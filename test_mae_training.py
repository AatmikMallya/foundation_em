import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

# Import our modules
from vit_3d import mae_vit_3d_small, get_device
from synthetic_data import create_synthetic_dataloader, SyntheticEMDataset

def train_mae_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train MAE for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, volumes in enumerate(progress_bar):
        volumes = volumes.to(device)
        
        # Forward pass
        loss, pred, mask = model(volumes)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    return total_loss / num_batches

def visualize_reconstruction(model, dataloader, device, save_path=None):
    """Visualize MAE reconstruction on a sample."""
    model.eval()
    
    with torch.no_grad():
        # Get a batch
        volumes = next(iter(dataloader)).to(device)
        loss, pred, mask = model(volumes)
        
        # Take first sample
        volume = volumes[0, 0].cpu().numpy()  # Remove batch and channel dims
        pred_patches = pred[0].cpu().numpy()  # Predictions for first sample
        mask_tokens = mask[0].cpu().numpy()   # Mask for first sample
        
        # Reconstruct the volume
        reconstructed_volume = model.unpatchify(pred.cpu())
        reconstructed_volume = reconstructed_volume[0, 0].numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original volume - different slices
        slice_idx = volume.shape[0] // 2
        axes[0, 0].imshow(volume[slice_idx], cmap='gray')
        axes[0, 0].set_title('Original (Z-slice)')
        axes[0, 0].axis('off')
        
        slice_idx = volume.shape[1] // 2
        axes[0, 1].imshow(volume[:, slice_idx], cmap='gray')
        axes[0, 1].set_title('Original (Y-slice)')
        axes[0, 1].axis('off')
        
        slice_idx = volume.shape[2] // 2
        axes[0, 2].imshow(volume[:, :, slice_idx], cmap='gray')
        axes[0, 2].set_title('Original (X-slice)')
        axes[0, 2].axis('off')
        
        # Reconstructed volume
        slice_idx = reconstructed_volume.shape[0] // 2
        axes[1, 0].imshow(reconstructed_volume[slice_idx], cmap='gray')
        axes[1, 0].set_title('Reconstructed (Z-slice)')
        axes[1, 0].axis('off')
        
        slice_idx = reconstructed_volume.shape[1] // 2
        axes[1, 1].imshow(reconstructed_volume[:, slice_idx], cmap='gray')
        axes[1, 1].set_title('Reconstructed (Y-slice)')
        axes[1, 1].axis('off')
        
        slice_idx = reconstructed_volume.shape[2] // 2
        axes[1, 2].imshow(reconstructed_volume[:, :, slice_idx], cmap='gray')
        axes[1, 2].set_title('Reconstructed (X-slice)')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'MAE Reconstruction (Loss: {loss.item():.4f}, Mask Ratio: {mask.mean().item():.2f})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()

def main():
    """Main training function."""
    print("Testing MAE ViT-3D on Synthetic EM Data")
    print("=" * 50)
    
    # Configuration
    config = {
        'batch_size': 4,
        'num_epochs': 5,
        'learning_rate': 1e-4,
        'volume_size': (32, 32, 32),
        'patch_size': (8, 8, 8),
        'num_train_samples': 100,
        'num_val_samples': 20,
        'mask_ratio': 0.75,
    }
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nCreating synthetic datasets...")
    train_dataloader = create_synthetic_dataloader(
        batch_size=config['batch_size'],
        num_samples=config['num_train_samples'],
        volume_size=config['volume_size'],
        shuffle=True
    )
    
    val_dataloader = create_synthetic_dataloader(
        batch_size=config['batch_size'],
        num_samples=config['num_val_samples'],
        volume_size=config['volume_size'],
        shuffle=False
    )
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")
    
    # Create model
    print("\nCreating MAE model...")
    model = mae_vit_3d_small(
        volume_size=config['volume_size'],
        patch_size=config['patch_size'],
        mask_ratio=config['mask_ratio']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.05)
    
    # Training loop
    print("\nStarting training...")
    train_losses = []
    
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        # Train
        train_loss = train_mae_one_epoch(model, train_dataloader, optimizer, device, epoch + 1)
        train_losses.append(train_loss)
        
        print(f"Epoch {epoch + 1}/{config['num_epochs']}: Train Loss = {train_loss:.4f}")
        
        # Validation visualization every few epochs
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print("Generating reconstruction visualization...")
            try:
                visualize_reconstruction(
                    model, val_dataloader, device, 
                    save_path=f"mae_reconstruction_epoch_{epoch+1}.png"
                )
            except Exception as e:
                print(f"Visualization failed: {e}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MAE Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curve.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test intermediate features (for future SAE training)
    print("\nTesting intermediate feature extraction...")
    model.eval()
    with torch.no_grad():
        test_volume = next(iter(val_dataloader))[:1].to(device)  # Single sample
        
        # Test different layers
        for layer_idx in [2, 4, 6]:
            if layer_idx < len(model.encoder.blocks):
                features = model.encoder.get_intermediate_features(test_volume, layer_idx)
                print(f"Layer {layer_idx} features shape: {features.shape}")
    
    print("\nMAE training test completed successfully! ✅")
    print("\nNext steps for your PhD project:")
    print("1. Replace synthetic data with real hemibrain data using util_files/voxel_utils.py")
    print("2. Scale up to ViT-B size and larger volumes")
    print("3. Train on 2 teravoxels of hemibrain data")
    print("4. Extract layer-6 features for SAE training")
    
    return model, train_losses

if __name__ == "__main__":
    # Test the training pipeline
    model, losses = main() 