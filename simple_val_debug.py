#!/usr/bin/env python3
"""
Simple validation debug - no external deps
"""
import torch
import sys
import tarfile
import numpy as np
from pathlib import Path

sys.path.append('.')
from vit_3d import mae_vit_3d_base

def simple_validation_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load a few volumes manually from first shard
    shard_path = Path("/gpfs/radev/home/am3833/scratch/volumes_96/shard_00000.tar")
    
    volumes = []
    print("Loading volumes from shard...")
    
    with tarfile.open(shard_path, "r|") as tar:
        for i, member in enumerate(tar):
            if i >= 8:  # Load 8 volumes
                break
            buf = tar.extractfile(member).read()
            vol = np.frombuffer(buf, np.float32).reshape(96, 96, 96)
            volumes.append(torch.from_numpy(vol).unsqueeze(0))  # Add channel dim
    
    # Stack into batch
    batch = torch.stack(volumes).to(device)
    print(f"Batch shape: {batch.shape}")
    print(f"Batch range: [{batch.min():.3f}, {batch.max():.3f}]")
    
    # Create model
    model = mae_vit_3d_base(
        volume_size=(96, 96, 96),
        patch_size=16,
        norm_pix_loss=False,
        mask_ratio=0.5
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test validation-style forward pass
    print("\nTesting validation-style forward pass...")
    model.eval()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            loss, pred, mask, stats = model(batch, mask_ratio=0.5)
        
        print(f"Loss: {loss.item()}")
        print(f"Loss is NaN: {torch.isnan(loss)}")
        print(f"Pred shape: {pred.shape if pred is not None else 'None'}")
        print(f"Pred has NaN: {torch.isnan(pred).any() if pred is not None else 'No pred'}")
        print(f"Mask shape: {mask.shape if mask is not None else 'None'}")
        print(f"Mask sum (masked tokens): {mask.sum().item() if mask is not None else 'None'}")
        
        if mask is not None:
            mask_ratio_actual = mask.sum().item() / mask.numel()
            print(f"Actual mask ratio: {mask_ratio_actual:.3f}")
    
    # Test training-style forward pass
    print("\nTesting training-style forward pass...")
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast(enabled=True):
        loss, pred, mask, stats = model(batch, mask_ratio=0.5)
    
    print(f"Training loss: {loss.item()}")
    print(f"Training loss is NaN: {torch.isnan(loss)}")
    
    if not torch.isnan(loss):
        print("Testing backward pass...")
        scaler.scale(loss).backward()
        
        # Check SCALED gradients first (what I was measuring before)
        grad_norms_scaled = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms_scaled.append(param.grad.norm().item())
        
        total_scaled_norm = sum(g**2 for g in grad_norms_scaled)**0.5
        print(f"Current loss scale: {scaler.get_scale()}")
        print(f"Total gradient norm (SCALED): {total_scaled_norm:.3f}")
        
        # Now UNSCALE the gradients to get real values
        scaler.unscale_(optimizer)
        
        # Check UNSCALED gradients  
        grad_norms = []
        nan_grads = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if torch.isnan(param.grad).any():
                    nan_grads += 1
                    print(f"  NaN gradient in: {name}")
                if torch.isinf(param.grad).any():
                    print(f"  Inf gradient in: {name}")
        
        total_grad_norm = sum(g**2 for g in grad_norms)**0.5
        print(f"Total gradient norm (UNSCALED): {total_grad_norm:.3f}")
        print(f"Ratio (scaled/unscaled): {total_scaled_norm/total_grad_norm:.1f}")
        print(f"Parameters with NaN gradients: {nan_grads}")
        
        if nan_grads == 0:
            scaler.step(optimizer)
            scaler.update()
            print("Optimizer step completed successfully")
        else:
            print("Skipped optimizer step due to NaN gradients")

if __name__ == "__main__":
    simple_validation_test() 