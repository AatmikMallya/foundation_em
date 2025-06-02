import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Import our modules
from vit_3d import mae_vit_3d_small, get_device
from simple_synthetic_data import SimpleSyntheticDataset

def test_model_components():
    """Test individual model components."""
    print("Testing Model Components")
    print("=" * 40)
    
    device = get_device()
    volume_size = (32, 32, 32)
    patch_size = (8, 8, 8)
    
    # Create model
    model = mae_vit_3d_small(
        volume_size=volume_size,
        patch_size=patch_size,
        mask_ratio=0.0  # No masking first
    ).to(device)
    
    # Create simple test data
    test_vol = torch.zeros(1, 1, 32, 32, 32).to(device)
    # Add a simple sphere
    center = 16
    radius = 6
    for i in range(32):
        for j in range(32):
            for k in range(32):
                if (i-center)**2 + (j-center)**2 + (k-center)**2 <= radius**2:
                    test_vol[0, 0, i, j, k] = 1.0
    
    print(f"Test volume stats: min={test_vol.min():.3f}, max={test_vol.max():.3f}, mean={test_vol.mean():.3f}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        loss, pred, mask = model(test_vol)
        reconstructed = model.unpatchify(pred)
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Pred shape: {pred.shape}")
        print(f"Mask mean: {mask.mean().item():.3f}")
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Reconstructed stats: min={reconstructed.min():.3f}, max={reconstructed.max():.3f}, mean={reconstructed.mean():.3f}")
        
        # Calculate correlation
        original = test_vol[0, 0].cpu().numpy().flatten()
        recon = reconstructed[0, 0].cpu().numpy().flatten()
        correlation = np.corrcoef(original, recon)[0, 1]
        print(f"Correlation with no masking: {correlation:.4f}")
        
        if correlation > 0.5:
            print("✅ Model can reconstruct without masking")
        else:
            print("❌ Model cannot reconstruct even without masking")
    
    return model, test_vol

def test_different_mask_ratios():
    """Test reconstruction with different mask ratios."""
    print("\n\nTesting Different Mask Ratios")
    print("=" * 40)
    
    device = get_device()
    volume_size = (32, 32, 32)
    patch_size = (8, 8, 8)
    
    # Create simple test data
    test_vol = torch.zeros(1, 1, 32, 32, 32).to(device)
    center = 16
    radius = 6
    for i in range(32):
        for j in range(32):
            for k in range(32):
                if (i-center)**2 + (j-center)**2 + (k-center)**2 <= radius**2:
                    test_vol[0, 0, i, j, k] = 1.0
    
    mask_ratios = [0.0, 0.25, 0.5, 0.75]
    
    for mask_ratio in mask_ratios:
        print(f"\nTesting mask ratio: {mask_ratio}")
        
        model = mae_vit_3d_small(
            volume_size=volume_size,
            patch_size=patch_size,
            mask_ratio=mask_ratio
        ).to(device)
        model.eval()
        with torch.no_grad():
            loss, pred, mask = model(test_vol)
            reconstructed = model.unpatchify(pred)
            
            original = test_vol[0, 0].cpu().numpy().flatten()
            recon = reconstructed[0, 0].cpu().numpy().flatten()
            correlation = np.corrcoef(original, recon)[0, 1]
            
            print(f"  Loss: {loss.item():.4f}, Correlation: {correlation:.4f}")

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
        radius = d // 4
        for i in range(d):
            for j in range(h):
                for k in range(w):
                    if (i-center)**2 + (j-center)**2 + (k-center)**2 <= radius**2:
                        volume_np[i, j, k] = 1.0
        return torch.from_numpy(volume_np).unsqueeze(0) # Add channel dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sphere_volume

def test_simple_training():
    """Test training with minimal masking on a fixed shape."""
    print("\n\nTesting Simple Training (No Masking, Fixed Shape)")
    print("=" * 50)
    
    device = get_device()
    volume_size = (32, 32, 32)
    patch_size = (8, 8, 8)
    
    # Create model with no masking
    model = mae_vit_3d_small(
        volume_size=volume_size,
        patch_size=patch_size,
        mask_ratio=0.0  # No masking
    ).to(device)
    
    # Create fixed sphere dataloader
    fixed_dataset = FixedSphereDataset(num_samples=100, volume_size=volume_size)
    dataloader = DataLoader(fixed_dataset, batch_size=8, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05) # Lower LR, added weight decay
    
    print("Training for 30 epochs with no masking on a fixed sphere...")
    losses = []
    
    for epoch in range(30): # Increased epochs
        epoch_loss = 0
        num_batches = 0
        model.train()
        for volumes in dataloader:
            volumes = volumes.to(device)
            
            optimizer.zero_grad()
            loss, pred, mask = model(volumes)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Test correlation mid-training
        model.eval()
        with torch.no_grad():
            test_vol_eval = fixed_dataset.sphere_volume.unsqueeze(0).to(device)
            loss_eval, pred_eval, mask_eval = model(test_vol_eval)
            reconstructed_eval = model.unpatchify(pred_eval)
            original_eval = test_vol_eval[0, 0].cpu().numpy().flatten()
            recon_eval = reconstructed_eval[0, 0].cpu().numpy().flatten()
            correlation_eval = np.corrcoef(original_eval, recon_eval)[0, 1]

        print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.5f}, Correlation = {correlation_eval:.4f}")
    
    # Test final performance
    print("\nTesting final performance on the fixed sphere...")
    model.eval()
    with torch.no_grad():
        test_vol = fixed_dataset.sphere_volume.unsqueeze(0).to(device)
        loss, pred, mask = model(test_vol)
        reconstructed = model.unpatchify(pred)
        
        original = test_vol[0, 0].cpu().numpy().flatten()
        recon = reconstructed[0, 0].cpu().numpy().flatten()
        correlation = np.corrcoef(original, recon)[0, 1]
        
        print(f"Final correlation: {correlation:.4f}")
        
        if correlation > 0.8:
            print("✅ Model learned to reconstruct the fixed sphere without masking!")
        elif correlation > 0.5:
            print("⚠️ Model somewhat learned to reconstruct the fixed sphere.")
        else:
            print("❌ Model failed to learn the fixed sphere even without masking")
    
    return losses, correlation # Return final correlation

def main():
    """Run all debugging tests."""
    print("MAE Debugging Session - Phase 2 (Focus on Basic Learning)")
    print("=" * 60)
    
    # Test 1: Basic model components (already informative)
    # model, test_vol = test_model_components()
    
    # Test 2: Different mask ratios (already informative)
    # test_different_mask_ratios()
    
    # Test 3: Simple training on a fixed shape
    losses, final_correlation = test_simple_training()
    
    # Plot training curve
    if losses:
        plt.figure(figsize=(10, 6))
        plt.plot(losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss (No Masking, Fixed Sphere)')
        plt.grid(True)
        plt.savefig("debug_fixed_sphere_training_curve.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    print("\n" + "=" * 60)
    print("DEBUGGING SUMMARY - PHASE 2:")
    print("=" * 60)
    
    print(f"Final correlation on fixed sphere (no masking): {final_correlation:.4f}")
    if final_correlation > 0.8:
        print("✅ Basic reconstruction learning SUCCESSFUL.")
    elif final_correlation > 0.5:
        print("⚠️ Basic reconstruction learning PARTIALLY SUCCESSFUL.")
    else:
        print("❌ Basic reconstruction learning FAILED.")

    if losses and len(losses) > 1:
        improvement = losses[0] - losses[-1]
        print(f"Loss improvement: {improvement:.5f}")
        if improvement > 0.001: # Adjusted threshold for smaller losses
            print("✅ Model is capable of reducing loss on this simple task.")
        else:
            print("❌ Model has difficulty reducing loss even on this simple task.")

if __name__ == "__main__":
    main() 