#!/usr/bin/env python3
"""
Debug validation loss of 0
"""
import torch
import sys
from pathlib import Path

sys.path.append('.')
from vol_train import TarShardDataset, run_val
from torch.utils.data import DataLoader
from vit_3d import mae_vit_3d_base

def debug_validation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup exactly like training
    shards = sorted(Path("/gpfs/radev/home/am3833/scratch/volumes_96").glob("shard*.tar"))[:1]
    
    val_loader = DataLoader(
        TarShardDataset(shards, 96, shuffle=False),
        batch_size=8,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    model = mae_vit_3d_base(
        volume_size=(96, 96, 96),
        patch_size=16,
        norm_pix_loss=False,
        mask_ratio=0.5
    ).to(device)
    
    print("Testing validation behavior...")
    
    # Test the validation function
    val_loss = run_val(val_loader, model, 0.5, True, device)
    print(f"Validation loss from run_val(): {val_loss}")
    
    # Manual validation to see what's happening
    model.eval()
    losses = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 3:  # Just test a few batches
                break
                
            batch = batch.to(device, non_blocking=True)
            print(f"Batch {i}: shape={batch.shape}, range=[{batch.min():.3f}, {batch.max():.3f}]")
            
            with torch.cuda.amp.autocast(enabled=True):
                loss, pred, mask, stats = model(batch, mask_ratio=0.5)
            
            print(f"  Loss: {loss.item()}")
            print(f"  Pred shape: {pred.shape if pred is not None else 'None'}")
            print(f"  Mask sum: {mask.sum().item() if mask is not None else 'None'}")
            print(f"  NaN in loss: {torch.isnan(loss)}")
            print(f"  NaN in pred: {torch.isnan(pred).any() if pred is not None else 'No pred'}")
            
            if not torch.isnan(loss):
                losses.append(loss.detach())
    
    if losses:
        avg_loss = torch.stack(losses).mean()
        print(f"\nManual validation average: {avg_loss.item()}")
    else:
        print("\nNo valid losses collected!")

if __name__ == "__main__":
    debug_validation() 