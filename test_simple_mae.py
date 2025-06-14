import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import wandb # Import wandb

# Import our modules
from vit_3d import mae_vit_3d_small, get_device
from simple_synthetic_data import create_simple_dataloader, SimpleSyntheticDataset

def train_mae_one_epoch(model, dataloader, optimizer, device, epoch, mask_ratio):
    """Train MAE for one epoch with detailed monitoring."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} (Mask {mask_ratio*100:.0f}%)")
    
    for batch_idx, volumes in enumerate(progress_bar):
        volumes = volumes.to(device)
        
        # Forward pass with current epoch's mask_ratio
        loss, pred, mask, _ = model(volumes, mask_ratio=mask_ratio)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0: # Update less frequently
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate_mae(model, dataloader, device, mask_ratio):
    """Evaluate MAE on validation data."""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for volumes in dataloader:
            volumes = volumes.to(device)
            loss, pred, mask, _ = model(volumes, mask_ratio=mask_ratio)
            total_loss += loss.item()
            
    return total_loss / num_batches

def visualize_reconstruction_detailed(model, dataloader, device, epoch, mask_ratio, save_path_prefix="simple_mae_reconstruction"):
    """Detailed visualization with side-by-side comparison."""
    model.eval()
    save_path = None # Initialize save_path
    
    with torch.no_grad():
        volumes = next(iter(dataloader)).to(device)
        # Use a fixed mask ratio for consistent visualization if needed, or current epoch's
        loss, pred, mask, _ = model(volumes, mask_ratio=mask_ratio) 
        
        original_volume = volumes[0, 0].cpu().numpy()
        reconstructed_volume_patches = model.unpatchify(pred.cpu())
        reconstructed_volume = reconstructed_volume_patches[0, 0].numpy()
        
        mse = np.mean((original_volume - reconstructed_volume) ** 2)
        mae = np.mean(np.abs(original_volume - reconstructed_volume))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10)) # Reduced to 2 rows for key slices
        slice_indices = [original_volume.shape[0] // 4, original_volume.shape[0] // 2, 3 * original_volume.shape[0] // 4]

        for i, slice_idx in enumerate(slice_indices):
            ax_orig = axes[0, i]
            ax_recon = axes[1, i]
            
            im_orig = ax_orig.imshow(original_volume[slice_idx], cmap='gray', vmin=0, vmax=1)
            ax_orig.set_title(f'Original Z={slice_idx}')
            ax_orig.axis('off')

            im_recon = ax_recon.imshow(reconstructed_volume[slice_idx], cmap='gray', vmin=0, vmax=1)
            ax_recon.set_title(f'Reconstructed Z={slice_idx}')
            ax_recon.axis('off')

        plt.suptitle(f'Epoch {epoch} (Mask {mask_ratio*100:.0f}%) - Loss: {loss.item():.4f}, MSE: {mse:.4f}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        
        save_path = f"{save_path_prefix}_epoch_{epoch}_mask_{int(mask_ratio*100)}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Detailed visualization saved to {save_path}")
        plt.close(fig) # Close the figure to free memory
    return save_path # Return the path for wandb logging

def check_learning_progress(model, device, mask_ratio_eval=0.75):
    """Check model's ability to reconstruct a known simple volume (e.g., a sphere) 
       when it was trained with masking."""
    model.eval()
    correlation = 0.0
    mse_val = 0.0
    loss_val = 0.0

    with torch.no_grad():
        test_vol = torch.zeros(1, 1, 32, 32, 32).to(device) # Batch size 1, 1 channel
        center = 16
        radius = 6
        for i_idx in range(32):
            for j_idx in range(32):
                for k_idx in range(32):
                    if (i_idx-center)**2 + (j_idx-center)**2 + (k_idx-center)**2 <= radius**2:
                        test_vol[0, 0, i_idx, j_idx, k_idx] = 1.0
        
        loss, pred, mask, _ = model(test_vol, mask_ratio=mask_ratio_eval) # Evaluate with a fixed high mask ratio
        reconstructed = model.unpatchify(pred)
        
        original_sphere_flat = test_vol[0, 0].cpu().numpy().flatten()
        recon_sphere_flat = reconstructed[0, 0].cpu().numpy().flatten()
        
        if len(original_sphere_flat) == len(recon_sphere_flat) and np.std(original_sphere_flat) > 1e-6 and np.std(recon_sphere_flat) > 1e-6:
            correlation = np.corrcoef(original_sphere_flat, recon_sphere_flat)[0, 1]
        else:
            correlation = 0 # Avoid NaN if variance is zero

        mse_val = np.mean((original_sphere_flat - recon_sphere_flat)**2)
        loss_val = loss.item()
        
        print(f"  Sphere Test (Mask {mask_ratio_eval*100:.0f}%): Loss={loss_val:.4f}, MSE={mse_val:.4f}, Corr={correlation:.4f}")
    return mse_val, correlation

def main():
    print("Training MAE ViT-3D on Simple Geometric Shapes with Masking (with wandb)")
    print("=" * 70)
    
    config = {
        'batch_size': 16, 
        'num_epochs': 200, # Changed to 200 epochs
        'learning_rate': 1e-4,
        'volume_size': (32, 32, 32),
        'patch_size': (8, 8, 8),
        'num_train_samples': 2000, 
        'num_val_samples': 400,
        'initial_mask_ratio': 0.50, 
        'final_mask_ratio': 0.75, 
        'max_shapes_per_volume': 3, 
        'add_noise_level': None,  
        'data_seed': 42,          
        'mask_ratio_epoch_split': 100, # Adjusted for 200 epochs
        'num_workers': 0, 
    }
    
    # Initialize wandb
    wandb.init(project="mae-3d-simple-shapes", config=config, name="mae_simple_200epochs")
    
    device = get_device()
    print(f"Using device: {device}")
    wandb.config.update({"device": str(device)}) # Log device to wandb
    
    train_dataloader = create_simple_dataloader(
        batch_size=config['batch_size'],
        num_samples=config['num_train_samples'],
        volume_size=config['volume_size'],
        max_shapes_per_volume=config['max_shapes_per_volume'], 
        add_noise_level=config['add_noise_level'], 
        num_workers=config['num_workers'],
        shuffle=True,
        seed=config['data_seed']
    )
    val_dataloader = create_simple_dataloader(
        batch_size=config['batch_size'],
        num_samples=config['num_val_samples'],
        volume_size=config['volume_size'],
        max_shapes_per_volume=config['max_shapes_per_volume'], 
        add_noise_level=config['add_noise_level'], 
        num_workers=config['num_workers'],
        shuffle=False, 
        seed=config['data_seed'] + 1 if config['data_seed'] is not None else None 
    )
    
    model = mae_vit_3d_small(
        volume_size=config['volume_size'], 
        patch_size=config['patch_size'],
        mask_ratio=config['initial_mask_ratio']  # Set initial mask ratio properly
    ).to(device)
    wandb.watch(model, log_freq=100) # Log model gradients and parameters

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.05)

    print("\nStarting training...")
    current_mask_ratio = config['initial_mask_ratio']

    for epoch in range(config['num_epochs']):
        if epoch >= config['mask_ratio_epoch_split'] and config['final_mask_ratio'] > config['initial_mask_ratio']:
             current_mask_ratio = config['final_mask_ratio']

        train_loss = train_mae_one_epoch(model, train_dataloader, optimizer, device, epoch + 1, current_mask_ratio)
        val_loss = evaluate_mae(model, val_dataloader, device, current_mask_ratio)
        
        _, sphere_corr = check_learning_progress(model, device, mask_ratio_eval=0.75)
        
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']:3d} (Mask {current_mask_ratio*100:.0f}%): Tr Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Sphere Corr={sphere_corr:.3f}")
        
        # Log metrics to wandb
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "sphere_correlation": sphere_corr,
            "mask_ratio": current_mask_ratio,
            "learning_rate": optimizer.param_groups[0]['lr'] # Log current learning rate
        }

        vis_path = None
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == config['num_epochs'] - 1: # Visualize every 10 epochs
            print(f"  Generating reconstruction visualization for epoch {epoch + 1}...")
            try:
                vis_path = visualize_reconstruction_detailed(model, val_dataloader, device, epoch + 1, current_mask_ratio)
                if vis_path:
                    log_dict["reconstruction_visual"] = wandb.Image(vis_path)
            except Exception as e:
                print(f"  Visualization failed: {e}")
        
        wandb.log(log_dict) # Log all metrics for the epoch
    
    # Plotting (simplified) - also log this to wandb
    epochs_range = range(1, config['num_epochs'] + 1)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    # Fetch logged data from wandb history for plotting if needed or use local lists
    # For simplicity, let's assume we still want to plot from local history if available or wandb data
    # We need to store train_losses, val_losses, sphere_corrs if we want to plot them directly
    # For now, relying on wandb for plotting, but keeping a local plot save.
    
    # If we want to reproduce the plot locally from wandb data (example, not fully implemented here for brevity)
    # history = wandb.run.history()
    # train_losses_wandb = history["train_loss"].tolist()
    # val_losses_wandb = history["val_loss"].tolist()
    # sphere_corrs_wandb = history["sphere_correlation"].tolist()
    # ax1.plot(epochs_range, train_losses_wandb, color=color, linestyle='-', label='Train Loss')
    # ax1.plot(epochs_range, val_losses_wandb, color=color, linestyle=':', label='Val Loss')
    
    # Fallback to placeholder if direct history access is complex for this snippet
    # This part of the script might need adjustment if train_losses etc. lists are not populated
    # For now, we are logging epoch by epoch, wandb handles the plotting.
    # Let's just save a placeholder or a message.
    # The following local plot will be empty as train_losses etc. are not stored in lists anymore.

    ax1.tick_params(axis='y', labelcolor=color)
    # ax2 = ax1.twinx() 
    # color = 'tab:blue'
    # ax2.set_ylabel('Correlation', color=color)  
    # ax2.plot(epochs_range, sphere_corrs_wandb, color=color, linestyle='--', label='Sphere Recon Correlation')
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_ylim(0, 1)
    
    plt.title('Training Metrics (View in WandB for full history)')
    plt.grid(True)
    fig.tight_layout() 
    plot_save_path = "simple_mae_training_curves_summary.png"
    plt.savefig(plot_save_path, dpi=150)
    print(f"Summary plot saved to {plot_save_path}")
    wandb.log({"final_summary_plot": wandb.Image(plot_save_path)})
    plt.close(fig)

    print("\nTraining with masking and wandb logging completed.")
    # Final status based on the last sphere_corr logged to wandb or kept in a variable
    # last_sphere_corr = sphere_corrs_wandb[-1] if sphere_corrs_wandb else 0
    last_sphere_corr = sphere_corr # From the last epoch
    if last_sphere_corr > 0.5:
        print(f"✅ Model shows good reconstruction capability (Corr: {last_sphere_corr:.3f}) with masking!")
    elif last_sphere_corr > 0.2:
        print(f"⚠️ Model shows some reconstruction capability (Corr: {last_sphere_corr:.3f}) with masking.")
    else:
        print(f"❌ Model struggles with reconstruction (Corr: {last_sphere_corr:.3f}) with masking.")
    
    wandb.finish()

if __name__ == "__main__":
    main() 