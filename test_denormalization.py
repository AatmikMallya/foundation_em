#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from vit_3d import mae_vit_3d_small, get_device

def test_denormalization():
    """Test that denormalization produces coherent reconstructions."""
    print("Testing Denormalization with norm_pix_loss")
    print("=" * 50)
    
    device = get_device()
    volume_size = (32, 32, 32)
    patch_size = (8, 8, 8)
    
    # Create a simple test volume with clear structure
    test_vol = torch.zeros(1, 1, 32, 32, 32).to(device)
    
    # Add a sphere
    center = 16
    radius = 6
    for i in range(32):
        for j in range(32):
            for k in range(32):
                if (i-center)**2 + (j-center)**2 + (k-center)**2 <= radius**2:
                    test_vol[0, 0, i, j, k] = 1.0
    
    # Add a cube in another region
    for i in range(8, 16):
        for j in range(8, 16):
            for k in range(8, 16):
                test_vol[0, 0, i, j, k] = 0.7
    
    print(f"Test volume stats: min={test_vol.min():.3f}, max={test_vol.max():.3f}, mean={test_vol.mean():.3f}")
    
    # Test with norm_pix_loss enabled
    model_with_norm = mae_vit_3d_small(
        volume_size=volume_size,
        patch_size=patch_size,
        norm_pix_loss=True  # Enable normalization
    ).to(device)
    
    # Test with norm_pix_loss disabled
    model_without_norm = mae_vit_3d_small(
        volume_size=volume_size,
        patch_size=patch_size,
        norm_pix_loss=False  # Disable normalization
    ).to(device)
    
    # Copy weights to make fair comparison
    model_without_norm.load_state_dict(model_with_norm.state_dict())
    
    print("\nTesting with norm_pix_loss=True:")
    model_with_norm.eval()
    with torch.no_grad():
        loss_norm, pred_norm, mask_norm, patch_stats = model_with_norm(test_vol, mask_ratio=0.5)
        
        # Test denormalization
        pred_denormalized = pred_norm.cpu()
        if patch_stats is not None:
            mean, var = patch_stats
            pred_denormalized = pred_denormalized * (var.add(1.e-6).sqrt()).cpu() + mean.cpu()
            print("  Denormalization applied successfully")
        else:
            print("  No patch stats returned (this shouldn't happen with norm_pix_loss=True)")
        
        reconstructed_norm = model_with_norm.unpatchify(pred_denormalized)
        
        print(f"  Loss: {loss_norm.item():.4f}")
        print(f"  Reconstructed stats: min={reconstructed_norm.min():.3f}, max={reconstructed_norm.max():.3f}, mean={reconstructed_norm.mean():.3f}")
        
        # Calculate correlation
        original_flat = test_vol[0, 0].cpu().numpy().flatten()
        recon_flat = reconstructed_norm[0, 0].numpy().flatten()
        correlation_norm = np.corrcoef(original_flat, recon_flat)[0, 1]
        print(f"  Correlation: {correlation_norm:.4f}")
    
    print("\nTesting with norm_pix_loss=False:")
    model_without_norm.eval()
    with torch.no_grad():
        loss_no_norm, pred_no_norm, mask_no_norm, patch_stats_none = model_without_norm(test_vol, mask_ratio=0.5)
        reconstructed_no_norm = model_without_norm.unpatchify(pred_no_norm.cpu())
        
        print(f"  Loss: {loss_no_norm.item():.4f}")
        print(f"  Reconstructed stats: min={reconstructed_no_norm.min():.3f}, max={reconstructed_no_norm.max():.3f}, mean={reconstructed_no_norm.mean():.3f}")
        print(f"  Patch stats returned: {patch_stats_none is not None}")
        
        # Calculate correlation
        recon_flat_no_norm = reconstructed_no_norm[0, 0].numpy().flatten()
        correlation_no_norm = np.corrcoef(original_flat, recon_flat_no_norm)[0, 1]
        print(f"  Correlation: {correlation_no_norm:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slice_idx = 16
    
    axes[0].imshow(test_vol[0, 0, slice_idx].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed_norm[0, 0, slice_idx].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'With norm_pix_loss\nCorr: {correlation_norm:.3f}')
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed_no_norm[0, 0, slice_idx].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Without norm_pix_loss\nCorr: {correlation_no_norm:.3f}')
    axes[2].axis('off')
    
    plt.suptitle('Denormalization Test Results')
    plt.tight_layout()
    plt.savefig('denormalization_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nTest Results:")
    if patch_stats is not None:
        print("✅ Patch statistics are correctly returned with norm_pix_loss=True")
    else:
        print("❌ Patch statistics not returned with norm_pix_loss=True")
    
    if abs(reconstructed_norm.min().item()) < 2.0 and abs(reconstructed_norm.max().item()) < 2.0:
        print("✅ Denormalized reconstruction has reasonable value range")
    else:
        print("❌ Denormalized reconstruction has unreasonable value range")
    
    if correlation_norm > 0.1:  # Even untrained should have some correlation
        print("✅ Denormalized reconstruction shows reasonable correlation")
    else:
        print("❌ Denormalized reconstruction shows poor correlation")

if __name__ == "__main__":
    test_denormalization() 