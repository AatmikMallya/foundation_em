import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import wandb

# Import our modules
from vit_3d import mae_vit_3d_small, get_device
from membrane_synthetic_data import create_membrane_dataloader # Updated import

def train_mae_one_epoch(model, dataloader, optimizer, device, epoch, mask_ratio):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} (Mask {mask_ratio*100:.0f}%)")
    
    for batch_idx, volumes in enumerate(progress_bar):
        volumes = volumes.to(device)
        loss, pred, mask = model(volumes, mask_ratio=mask_ratio)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % (num_batches // 5) == 0: # Update progress bar 5 times per epoch
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate_mae(model, dataloader, device, mask_ratio):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for volumes in dataloader:
            volumes = volumes.to(device)
            loss, pred, mask = model(volumes, mask_ratio=mask_ratio)
            total_loss += loss.item()
            
    return total_loss / num_batches

def visualize_reconstructions(model, dataloader, device, epoch, mask_ratio, dataset_name, num_examples=5, save_dir="membrane_visualizations"):
    """Visualize reconstructions for a given number of examples from a dataloader."""
    model.eval()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    examples_shown = 0
    all_originals = []
    all_reconstructed = []
    all_losses = []

    with torch.no_grad():
        for batch_volumes in dataloader:
            if examples_shown >= num_examples:
                break
            volumes_to_device = batch_volumes.to(device)
            loss_batch, pred_batch, _ = model(volumes_to_device, mask_ratio=mask_ratio)
            
            reconstructed_batch = model.unpatchify(pred_batch.cpu())
            original_batch = batch_volumes # Keep originals on CPU

            for i in range(volumes_to_device.size(0)):
                if examples_shown >= num_examples:
                    break
                all_originals.append(original_batch[i, 0].cpu().numpy())
                all_reconstructed.append(reconstructed_batch[i, 0].cpu().numpy())
                # Note: loss_batch is for the whole batch. 
                # For individual loss, we'd need to re-run or approximate.
                # For simplicity, we can report batch loss or an average.
                all_losses.append(loss_batch.item() / volumes_to_device.size(0)) # Approximate per-sample loss
                examples_shown += 1

    if not all_originals:
        print(f"No examples to visualize for {dataset_name}.")
        return None

    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 3 * num_examples))
    if num_examples == 1:
        axes = np.array([axes]) # Make it iterable for single example

    avg_mse_total = 0
    for i in range(examples_shown):
        original_np = all_originals[i]
        reconstructed_np = all_reconstructed[i]
        slice_idx = original_np.shape[0] // 2
        mse = np.mean((original_np - reconstructed_np)**2)
        avg_mse_total += mse

        axes[i, 0].imshow(original_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'{dataset_name.capitalize()} Original {i+1} (Z={slice_idx})')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(reconstructed_np[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'{dataset_name.capitalize()} Recon {i+1} (Z={slice_idx})\nMSE: {mse:.4f}')
        axes[i, 1].axis('off')
    
    avg_mse = avg_mse_total / examples_shown if examples_shown > 0 else 0
    plt.suptitle(f'Epoch {epoch} - {dataset_name.capitalize()} Reconstructions (Mask {mask_ratio*100:.0f}%) - Avg Batch Loss: {np.mean(all_losses):.4f}, Avg Img MSE: {avg_mse:.4f}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    save_path = Path(save_dir) / f"{dataset_name}_epoch_{epoch}_mask_{int(mask_ratio*100)}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to {save_path}")
    plt.close(fig)
    return str(save_path)

def main():
    config = {
        'run_name': 'mae_membrane_200epochs',
        'project_name': 'mae-3d-membranes',
        'batch_size': 8, # Can be smaller for larger volumes if memory is an issue
        'num_epochs': 200,
        'learning_rate': 1e-4,
        'volume_size': (64, 64, 64), # For membrane data
        'patch_size': (8, 8, 8),    # (64/8)^3 = 512 patches
        'num_train_samples': 2000,
        'num_val_samples': 400,
        'initial_mask_ratio': 0.50,
        'final_mask_ratio': 0.75,
        'mask_ratio_epoch_split': 100, # Half of num_epochs
        'num_workers': 0,
        'visualization_epoch_interval': 10,
        'num_examples_to_visualize': 5,
        # Membrane dataset specific parameters (can be tuned)
        'num_gaussians_range': (10, 20),
        'gaussian_strength_range': (-1.0, 1.0),
        'gaussian_sigma_range': (10.0, 20.0),
        'isovalue_center': 0.0,
        'membrane_band_width': 0.15,
        'noise_level': 0.02,
        'data_seed': 42
    }

    wandb.init(project=config['project_name'], name=config['run_name'], config=config)
    
    device = get_device()
    print(f"Using device: {device}")
    wandb.config.update({"device": str(device)}, allow_val_change=True)

    # Create DataLoaders
    train_dataloader = create_membrane_dataloader(
        batch_size=config['batch_size'],
        num_samples=config['num_train_samples'],
        volume_size=config['volume_size'],
        num_gaussians_range=config['num_gaussians_range'],
        gaussian_strength_range=config['gaussian_strength_range'],
        gaussian_sigma_range=config['gaussian_sigma_range'],
        isovalue_center=config['isovalue_center'],
        membrane_band_width=config['membrane_band_width'],
        noise_level=config['noise_level'],
        num_workers=config['num_workers'],
        shuffle=True,
        seed=config['data_seed']
    )
    val_dataloader = create_membrane_dataloader(
        batch_size=config['batch_size'], # Can use larger batch for val if desired, but keep same for apples-to-apples vis
        num_samples=config['num_val_samples'],
        volume_size=config['volume_size'],
        num_gaussians_range=config['num_gaussians_range'],
        gaussian_strength_range=config['gaussian_strength_range'],
        gaussian_sigma_range=config['gaussian_sigma_range'],
        isovalue_center=config['isovalue_center'],
        membrane_band_width=config['membrane_band_width'],
        noise_level=config['noise_level'],
        num_workers=config['num_workers'],
        shuffle=False, # No shuffle for val/test
        seed=config['data_seed'] + 1 if config['data_seed'] is not None else None
    )

    model = mae_vit_3d_small(
        volume_size=config['volume_size'],
        patch_size=config['patch_size']
    ).to(device)
    wandb.watch(model, log_freq=1000) # Log model gradients and parameters, adjust freq as needed

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.05)

    print(f"\nStarting training on membrane data for {config['num_epochs']} epochs...")
    current_mask_ratio = config['initial_mask_ratio']

    for epoch in range(config['num_epochs']):
        if epoch >= config['mask_ratio_epoch_split']:
            current_mask_ratio = config['final_mask_ratio']

        train_loss = train_mae_one_epoch(model, train_dataloader, optimizer, device, epoch + 1, current_mask_ratio)
        val_loss = evaluate_mae(model, val_dataloader, device, current_mask_ratio) # Evaluate with current mask ratio
        
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']:3d} (Mask {current_mask_ratio*100:.0f}%): Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mask_ratio": current_mask_ratio,
            "learning_rate": optimizer.param_groups[0]['lr']
        }

        if (epoch + 1) % config['visualization_epoch_interval'] == 0 or epoch == 0 or epoch == config['num_epochs'] - 1:
            print(f"  Generating reconstructions for epoch {epoch + 1}...")
            try:
                train_vis_path = visualize_reconstructions(model, train_dataloader, device, epoch + 1, current_mask_ratio, "train", config['num_examples_to_visualize'])
                if train_vis_path:
                    log_dict["train_reconstructions"] = wandb.Image(train_vis_path)
                
                val_vis_path = visualize_reconstructions(model, val_dataloader, device, epoch + 1, current_mask_ratio, "val", config['num_examples_to_visualize'])
                if val_vis_path:
                    log_dict["val_reconstructions"] = wandb.Image(val_vis_path)
            except Exception as e:
                print(f"  Visualization failed for epoch {epoch + 1}: {e}")
        
        wandb.log(log_dict)
    
    # Final summary plot is less critical if all metrics are in wandb
    # but can create one for local record if desired.
    # For now, rely on wandb's plotting capabilities.

    print("\nTraining with membrane data and wandb logging completed.")
    wandb.finish()

if __name__ == "__main__":
    main() 