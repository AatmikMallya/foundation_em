#!/usr/bin/env python3

import torch
from vit_3d import mae_vit_3d_small

def test_mask_ratio_fix():
    """Test that the mask_ratio bug is fixed."""
    
    print("🔧 Testing mask_ratio fix...")
    
    # Test 1: Model should store mask_ratio correctly
    model = mae_vit_3d_small(
        volume_size=(32, 32, 32),
        patch_size=(8, 8, 8),
        mask_ratio=0.6
    )
    
    assert hasattr(model, 'mask_ratio'), "❌ Model should have mask_ratio attribute"
    assert model.mask_ratio == 0.6, f"❌ Model mask_ratio should be 0.6, got {model.mask_ratio}"
    print("✅ Model correctly stores mask_ratio")
    
    # Test 2: Model should use stored mask_ratio when none provided
    device = torch.device('cpu')
    model = model.to(device)
    
    test_volume = torch.randn(2, 1, 32, 32, 32).to(device)
    
    # Call without mask_ratio - should use stored value
    loss1, pred1, mask1, stats1 = model(test_volume)
    mask_ratio_used1 = mask1.float().mean().item()
    
    # Call with explicit mask_ratio
    loss2, pred2, mask2, stats2 = model(test_volume, mask_ratio=0.8)
    mask_ratio_used2 = mask2.float().mean().item()
    
    print(f"📊 Test Results:")
    print(f"  Without explicit mask_ratio: {mask_ratio_used1:.3f} (should be ~0.6)")
    print(f"  With explicit mask_ratio=0.8: {mask_ratio_used2:.3f} (should be ~0.8)")
    
    # Allow some tolerance due to randomness in masking
    assert abs(mask_ratio_used1 - 0.6) < 0.1, f"❌ Mask ratio should be ~0.6, got {mask_ratio_used1}"
    assert abs(mask_ratio_used2 - 0.8) < 0.1, f"❌ Mask ratio should be ~0.8, got {mask_ratio_used2}"
    
    print("✅ Model correctly uses mask_ratio (both stored and explicit)")
    print("✅ mask_ratio bug is FIXED!")
    
    return True

if __name__ == "__main__":
    test_mask_ratio_fix() 