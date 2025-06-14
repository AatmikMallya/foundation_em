import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

from vit_3d import mae_vit_3d_small, get_device

# Custom Dataset that always returns the same simple sphere
class FixedSphereDataset(Dataset):
    def __init__(self, num_samples, volume_size=(32,32,32)):
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.sphere_volume = self._create_sphere_volume()

    def _create_sphere_volume(self):
        d, h, w = self.volume_size
        volume_np = np.zeros((d, h, w), dtype=np.float32)
        center = d // 2
        radius = d // 5 # Slightly smaller radius for clearer structure
        for i in range(d):
            for j in range(h):
                for k in range(w):
                    if (i-center)**2 + (j-center)**2 + (k-center)**2 <= radius**2:
                        volume_np[i, j, k] = 1.0
        # Add channel dimension
        return torch.from_numpy(volume_np).unsqueeze(0) 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sphere_volume

def visualize_fixed_sphere_reconstruction(model, sphere_volume, mask_ratio, epoch, save_path_prefix="fixed_sphere_mae"):
    model.eval()
    device = sphere_volume.device
    with torch.no_grad():
        # Ensure sphere_volume has a batch dimension if it doesn't
        if sphere_volume.ndim == 4: # (C, D, H, W)
            input_vol = sphere_volume.unsqueeze(0) # (B, C, D, H, W)
        else:
            input_vol = sphere_volume

        loss, pred_patches, mask, _ = model(input_vol, mask_ratio=mask_ratio)
        reconstructed_volume_tensor = model.unpatchify(pred_patches)
        
        original_np = input_vol[0, 0].cpu().numpy()
        reconstructed_np = reconstructed_volume_tensor[0, 0].cpu().numpy()
        mask_np = mask[0].cpu().numpy() # (num_patches_total)
        
        # Reshape mask to be visualizable (e.g., one slice of patches)
        # Assuming 4x4x4 patches for a 32x32x32 volume with 8x8x8 patch size
        num_patches_d = model.encoder.patch_embed.num_patches_d
        num_patches_h = model.encoder.patch_embed.num_patches_h
        num_patches_w = model.encoder.patch_embed.num_patches_w
        mask_grid = mask_np.reshape(num_patches_d, num_patches_h, num_patches_w)
        mask_slice_to_show = mask_grid[num_patches_d // 2, :, :]

        mse = np.mean((original_np - reconstructed_np)**2)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        slice_idx = original_np.shape[0] // 2

        axes[0].imshow(original_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Original Sphere (Z={slice_idx})')
        axes[0].axis('off')

        axes[1].imshow(mask_slice_to_show, cmap='Reds', vmin=0, vmax=1)
        axes[1].set_title(f'Mask Pattern (Center Slice of Patches)')
        axes[1].axis('off')

        axes[2].imshow(reconstructed_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'Reconstructed (Z={slice_idx})')
        axes[2].axis('off')

        plt.suptitle(f'Epoch {epoch} (Mask {mask_ratio*100:.0f}%) - Loss: {loss.item():.4f}, MSE: {mse:.4f}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = f"{save_path_prefix}_epoch_{epoch}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.close(fig)
    return mse

def main():
    print("Focused Test: MAE on a Single, Fixed, Masked Sphere")
    print("=" * 60)

    config = {
        'batch_size': 8, # Can be small as data is identical
        'num_epochs': 300, # More epochs for this simple task
        'learning_rate': 1e-4,
        'volume_size': (32, 32, 32),
        'patch_size': (8, 8, 8),
        'mask_ratio': 0.75, # Fixed high masking ratio
        'num_dataset_samples': 100 # Arbitrary, as data is identical
    }

    device = get_device()
    print(f"Using device: {device}")

    # Dataset and DataLoader for the fixed sphere
    fixed_sphere_dataset = FixedSphereDataset(config['num_dataset_samples'], config['volume_size'])
    # The sphere_volume is (C,D,H,W), dataloader will add batch dim
    train_dataloader = DataLoader(fixed_sphere_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Get a single copy of the sphere for consistent evaluation
    sphere_for_eval = fixed_sphere_dataset.sphere_volume.to(device)
    if sphere_for_eval.ndim == 4: # (C,D,H,W)
         sphere_for_eval_batched = sphere_for_eval.unsqueeze(0) # Make (B,C,D,H,W)
    else: # Should already be (B,C,D,H,W) if dataloader was used, but this is direct from dataset
         sphere_for_eval_batched = sphere_for_eval

    model = mae_vit_3d_small(
        volume_size=config['volume_size'],
        patch_size=config['patch_size'],
        # mask_ratio will be passed directly to model.forward()
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.05)

    print(f"\nTraining on a fixed sphere with {config['mask_ratio']*100:.0f}% masking...")
    train_losses = []
    correlations = []

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")

        for batch_volumes in progress_bar:
            # All volumes in the batch are identical (the fixed sphere)
            volumes = batch_volumes.to(device)
            
            optimizer.zero_grad()
            loss, pred_patches, mask, _ = model(volumes, mask_ratio=config['mask_ratio'])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.5f}'})
        
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)

        # Evaluate reconstruction quality on the fixed sphere
        model.eval()
        with torch.no_grad():
            eval_loss, eval_pred_patches, eval_mask, _ = model(sphere_for_eval_batched, mask_ratio=config['mask_ratio'])
            reconstructed_eval = model.unpatchify(eval_pred_patches)
            
            original_flat = sphere_for_eval_batched[0,0].cpu().numpy().flatten()
            recon_flat = reconstructed_eval[0,0].cpu().numpy().flatten()
            
            correlation = 0.0
            if np.std(original_flat) > 1e-6 and np.std(recon_flat) > 1e-6:
                correlation = np.corrcoef(original_flat, recon_flat)[0, 1]
            correlations.append(correlation)
        
        print(f"Epoch {epoch+1:2d}: Avg Loss={avg_epoch_loss:.5f}, Eval Corr={correlation:.4f}, Eval Loss={eval_loss.item():.4f}")

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == config['num_epochs'] -1:
            visualize_fixed_sphere_reconstruction(model, sphere_for_eval, config['mask_ratio'], epoch + 1)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_losses, color=color, linestyle='-', label='Avg Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Correlation', color=color)
    ax2.plot(correlations, color=color, linestyle='--', label='Sphere Recon Correlation')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')

    fig.tight_layout() 
    plt.title('Fixed Sphere MAE: Loss & Correlation')
    plt.savefig("fixed_sphere_mae_learning_curves.png", dpi=150)
    plt.show()

    print("\nFocused MAE test on fixed sphere completed.")
    final_corr = correlations[-1] if correlations else 0.0
    if final_corr > 0.8:
        print(f"✅ SUCCESS: Model learned to inpaint the fixed sphere very well (Corr: {final_corr:.4f})!")
    elif final_corr > 0.5:
        print(f"⚠️ PARTIAL SUCCESS: Model learned to inpaint the fixed sphere somewhat (Corr: {final_corr:.4f}).")
    else:
        print(f"❌ FAILURE: Model struggled to inpaint the fixed sphere (Corr: {final_corr:.4f}).")

if __name__ == "__main__":
    main() 