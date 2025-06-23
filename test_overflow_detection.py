#!/usr/bin/env python3
"""
Test overflow detection during training
"""
import torch
import sys
import tarfile
import numpy as np
from pathlib import Path

sys.path.append('.')
from vit_3d import mae_vit_3d_base

def test_overflow_detection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load volumes from shard
    shard_path = Path("/gpfs/radev/home/am3833/scratch/volumes_96/shard_00000.tar")
    
    volumes = []
    print("Loading volumes from shard...")
    
    with tarfile.open(shard_path, "r|") as tar:
        for i, member in enumerate(tar):
            if i >= 16:  # Load 16 volumes for bigger batch
                break
            buf = tar.extractfile(member).read()
            vol = np.frombuffer(buf, np.float32).reshape(96, 96, 96)
            volumes.append(torch.from_numpy(vol.copy()).unsqueeze(0))  # Copy to make writable
    
    batch = torch.stack(volumes).to(device)
    print(f"Batch shape: {batch.shape}")
    
    # Create model
    model = mae_vit_3d_base(
        volume_size=(96, 96, 96),
        patch_size=16,
        norm_pix_loss=False,
        mask_ratio=0.5
    ).to(device)
    
    # Setup training exactly like real training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.05
    )
    
    # Standard GradScaler (likely to overflow)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    model.train()
    
    print(f"\nTesting overflow detection over multiple steps...")
    
    for step in range(20):  # Test multiple steps to trigger overflow
        print(f"\n--- Step {step+1} ---")
        
        optimizer.zero_grad()
        
        print(f"Scale before backward: {scaler.get_scale()}")
        
        with torch.cuda.amp.autocast():
            loss, pred, mask, stats = model(batch, mask_ratio=0.5)
        
        print(f"Loss: {loss.item():.4f}")
        
        if torch.isnan(loss):
            print("❌ NaN loss detected!")
            break
        
        # Scale and backward
        scaler.scale(loss).backward()
        
        # CRITICAL: Unscale before checking/clipping
        scaler.unscale_(optimizer)
        
        # Check for overflow in gradients
        found_inf = False
        inf_params = []
        nan_params = []
        
        for name, p in model.named_parameters():
            if p.grad is not None:
                if torch.isinf(p.grad).any():
                    found_inf = True
                    inf_params.append(name)
                if torch.isnan(p.grad).any():
                    found_inf = True
                    nan_params.append(name)
        
        # Measure gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"Unscaled grad norm: {total_norm:.3f}")
        print(f"Overflow detected: {found_inf}")
        
        if found_inf:
            print(f"❌ OVERFLOW DETECTED at step {step+1}!")
            if inf_params:
                print(f"  Inf in: {inf_params[:3]}{'...' if len(inf_params) > 3 else ''}")
            if nan_params:
                print(f"  NaN in: {nan_params[:3]}{'...' if len(nan_params) > 3 else ''}")
            
            # Force scale reduction
            scaler.update(found_inf)
            print(f"  Scale after overflow: {scaler.get_scale()}")
        else:
            # Normal step
            scaler.step(optimizer)
            scaler.update()
            print(f"✅ Step completed successfully")
        
        # Add small noise to make training more realistic
        if step % 3 == 0:
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.add_(torch.randn_like(p) * 1e-8)

if __name__ == "__main__":
    test_overflow_detection() 