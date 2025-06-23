#!/usr/bin/env python3
"""
NaN Reproduction Test
====================
Reproduces the NaN issue by testing specific scenarios that occur during training.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append('.')
from vit_3d import mae_vit_3d_base, enable_nan_checking, set_global_step

def create_test_volume(size, pattern="random"):
    """Create test volumes with different patterns."""
    if pattern == "random":
        return torch.randn(1, 1, size, size, size)
    elif pattern == "high_variance":
        # High variance data that might cause numerical issues
        vol = torch.randn(1, 1, size, size, size) * 5.0
        return vol
    elif pattern == "extreme_values":
        # Data with extreme values near float16 limits
        vol = torch.randn(1, 1, size, size, size)
        vol = vol * 30000  # Near FP16 limits
        return vol
    elif pattern == "synthetic_membrane":
        # Synthetic membrane-like data
        vol = torch.zeros(1, 1, size, size, size)
        # Add some membrane-like structures
        center = size // 2
        for i in range(3):
            start = center - 10 + i * 7
            end = center + 10 + i * 7
            vol[0, 0, start:end, :, :] = 0.8
            vol[0, 0, :, start:end, :] = 0.8
            vol[0, 0, :, :, start:end] = 0.8
        # Add noise
        vol += torch.randn_like(vol) * 0.1
        return vol

def test_forward_pass(model, volume, mask_ratio=0.5, test_name=""):
    """Test forward pass and check for NaN."""
    print(f"\n=== Testing {test_name} ===")
    print(f"Volume shape: {volume.shape}")
    print(f"Volume stats: min={volume.min():.3f}, max={volume.max():.3f}, mean={volume.mean():.3f}, std={volume.std():.3f}")
    
    model.eval()
    with torch.no_grad():
        try:
            loss, pred, mask, stats = model(volume, mask_ratio=mask_ratio)
            if torch.isnan(loss):
                print(f"❌ {test_name}: Forward pass produced NaN loss")
                return False
            elif torch.isnan(pred).any():
                print(f"❌ {test_name}: Forward pass produced NaN predictions")
                return False
            else:
                print(f"✅ {test_name}: Forward pass OK (loss={loss.item():.4f})")
                return True
        except Exception as e:
            print(f"💥 {test_name}: Forward pass failed with exception: {e}")
            return False

def test_backward_pass(model, volume, mask_ratio=0.5, use_amp=True, test_name=""):
    """Test backward pass and check for NaN."""
    print(f"\n=== Testing {test_name} (Backward) ===")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    try:
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                loss, pred, mask, stats = model(volume, mask_ratio=mask_ratio)
        else:
            loss, pred, mask, stats = model(volume, mask_ratio=mask_ratio)
        
        if torch.isnan(loss):
            print(f"❌ {test_name}: Forward produced NaN loss")
            return False
            
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            # Check gradients before unscaling
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            if np.isnan(total_norm) or np.isinf(total_norm):
                print(f"❌ {test_name}: Gradients contain NaN/Inf (norm={total_norm})")
                return False
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # Check gradients
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        print(f"❌ {test_name}: Parameter gradients contain NaN")
                        return False
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            optimizer.step()
        
        print(f"✅ {test_name}: Backward pass OK (loss={loss.item():.4f}, grad_norm={total_norm:.3f})")
        return True
        
    except Exception as e:
        print(f"💥 {test_name}: Backward pass failed with exception: {e}")
        return False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test configurations
    configs = [
        {"size": 64, "name": "64³_baseline"},
        {"size": 96, "name": "96³_target"},
    ]
    
    patterns = [
        ("random", "Random normal data"),
        ("high_variance", "High variance data"),
        ("synthetic_membrane", "Synthetic membrane data"),
        ("extreme_values", "Extreme values near FP16 limits"),
    ]
    
    # Enable NaN debugging
    enable_nan_checking(True)
    set_global_step(1000)  # Simulate around the step where NaN occurs
    
    results = []
    
    for config in configs:
        size = config["size"]
        config_name = config["name"]
        
        print(f"\n{'='*60}")
        print(f"Testing {config_name}")
        print(f"{'='*60}")
        
        # Create model for this size
        model = mae_vit_3d_base(
            volume_size=(size, size, size),
            patch_size=16,
            norm_pix_loss=False,
            mask_ratio=0.5
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        config_results = {"config": config_name, "results": {}}
        
        for pattern_name, pattern_desc in patterns:
            volume = create_test_volume(size, pattern_name).to(device)
            
            # Test forward pass
            fwd_ok = test_forward_pass(model, volume, 0.5, f"{config_name}_{pattern_name}_forward")
            
            # Test backward pass with AMP
            bwd_amp_ok = test_backward_pass(model, volume, 0.5, True, f"{config_name}_{pattern_name}_backward_amp")
            
            # Test backward pass without AMP
            bwd_no_amp_ok = test_backward_pass(model, volume, 0.5, False, f"{config_name}_{pattern_name}_backward_no_amp")
            
            config_results["results"][pattern_name] = {
                "forward": fwd_ok,
                "backward_amp": bwd_amp_ok,
                "backward_no_amp": bwd_no_amp_ok
            }
        
        results.append(config_results)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for config_result in results:
        config_name = config_result["config"]
        print(f"\n{config_name}:")
        
        for pattern_name, pattern_results in config_result["results"].items():
            fwd = "✅" if pattern_results["forward"] else "❌"
            bwd_amp = "✅" if pattern_results["backward_amp"] else "❌"
            bwd_no_amp = "✅" if pattern_results["backward_no_amp"] else "❌"
            
            print(f"  {pattern_name:20} | Forward: {fwd} | Backward+AMP: {bwd_amp} | Backward-AMP: {bwd_no_amp}")
    
    # Check if 96³ has more failures than 64³
    size_96_failures = sum([
        not result for pattern_results in results[1]["results"].values() 
        for result in pattern_results.values()
    ])
    size_64_failures = sum([
        not result for pattern_results in results[0]["results"].values() 
        for result in pattern_results.values()
    ])
    
    print(f"\nFailure count:")
    print(f"  64³: {size_64_failures} failures")
    print(f"  96³: {size_96_failures} failures")
    
    if size_96_failures > size_64_failures:
        print(f"\n🔍 HYPOTHESIS CONFIRMED: 96³ volumes have more numerical issues than 64³")
    else:
        print(f"\n🤔 HYPOTHESIS NOT CONFIRMED: Both sizes behave similarly")

if __name__ == "__main__":
    main() 