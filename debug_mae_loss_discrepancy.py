#!/usr/bin/env python3

import torch
import numpy as np
from vit_3d import mae_vit_3d_small
from membrane_synthetic_data import create_membrane_dataloader

def debug_loss_discrepancy():
    """Debug the discrepancy between training loss and visualization MSE."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and data
    model = mae_vit_3d_small(
        volume_size=(64, 64, 64),
        patch_size=(8, 8, 8),
        mask_ratio=0.4
    ).to(device)
    
    dataloader = create_membrane_dataloader(
        batch_size=2,
        num_samples=10,
        volume_size=(64, 64, 64),
        num_gaussians_range=(5, 15),
        gaussian_sigma_range=(5, 15),
        noise_level=0.01,
        membrane_band_width=0.1,
        num_workers=0,
        shuffle=False,
        seed=42,
        num_additional_spheres_range=(2, 5),
        additional_sphere_radius_range=(2.0, 4.0)
    )
    
    print("\n🔍 DEBUGGING LOSS DISCREPANCY")
    print("=" * 50)
    
    with torch.no_grad():
        volumes = next(iter(dataloader)).to(device)
        
        # Get model output
        loss, pred, mask, patch_stats = model(volumes, mask_ratio=0.4)
        
        print(f"📊 Model Output Analysis:")
        print(f"  Training Loss: {loss.item():.6f}")
        print(f"  Input shape: {volumes.shape}")
        print(f"  Pred shape: {pred.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask ratio: {mask.float().mean().item():.3f}")
        
        # Analyze prediction values
        print(f"\n📈 Prediction Statistics:")
        print(f"  Pred min: {pred.min().item():.6f}")
        print(f"  Pred max: {pred.max().item():.6f}")
        print(f"  Pred mean: {pred.mean().item():.6f}")
        print(f"  Pred std: {pred.std().item():.6f}")
        
        # Analyze input values
        print(f"\n📈 Input Statistics:")
        print(f"  Input min: {volumes.min().item():.6f}")
        print(f"  Input max: {volumes.max().item():.6f}")
        print(f"  Input mean: {volumes.mean().item():.6f}")
        print(f"  Input std: {volumes.std().item():.6f}")
        
        # Manual loss calculation
        print(f"\n🧮 Manual Loss Calculation:")
        
        # 1. Patchify the target
        target_patches = model.patchify(volumes)
        print(f"  Target patches shape: {target_patches.shape}")
        
        # 2. Calculate loss on masked patches only
        if patch_stats is not None:
            mean, var = patch_stats
            target_patches = (target_patches - mean) / (var + 1.e-6).sqrt()
            print(f"  Using normalized patch loss")
        
        # 3. Compute MSE only on masked patches
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)
        masked_pred = pred[mask_expanded == 1]
        masked_target = target_patches[mask_expanded == 1]
        
        manual_loss = torch.mean((masked_pred - masked_target) ** 2)
        print(f"  Manual MSE loss: {manual_loss.item():.6f}")
        print(f"  Model loss: {loss.item():.6f}")
        print(f"  Difference: {abs(manual_loss.item() - loss.item()):.6f}")
        
        # 4. Now calculate visualization-style MSE
        print(f"\n🖼️ Visualization-Style MSE Calculation:")
        
        # Denormalize predictions if needed
        pred_for_unpatchify = pred.cpu()
        if patch_stats is not None:
            mean_cpu, var_cpu = patch_stats[0].cpu(), patch_stats[1].cpu()
            pred_for_unpatchify = pred_for_unpatchify * (var_cpu + 1.e-6).sqrt() + mean_cpu
            print(f"  Denormalizing predictions")
        
        # Unpatchify to get full reconstruction
        reconstructed = model.unpatchify(pred_for_unpatchify)
        
        # Calculate MSE on full volume (this is what visualization shows)
        original_np = volumes[0, 0].cpu().numpy()
        reconstructed_np = reconstructed[0, 0].numpy()
        
        full_mse = np.mean((original_np - reconstructed_np) ** 2)
        print(f"  Full volume MSE: {full_mse:.6f}")
        
        # Calculate MSE only on visible patches (showing why it's low)
        # Create truth-masked reconstruction
        original_patches = model.patchify(volumes).cpu()
        mask_cpu = mask.cpu()
        
        # Combine: original visible + predicted masked
        mask_expanded_cpu = mask_cpu.unsqueeze(-1).expand_as(original_patches)
        if patch_stats is not None:
            # Need to denormalize original patches too for fair comparison
            original_patches_denorm = original_patches * (var_cpu + 1.e-6).sqrt() + mean_cpu
            combined_patches = torch.where(mask_expanded_cpu == 0, original_patches_denorm, pred_for_unpatchify)
        else:
            combined_patches = torch.where(mask_expanded_cpu == 0, original_patches, pred_for_unpatchify)
        
        truth_masked_recon = model.unpatchify(combined_patches)
        truth_masked_np = truth_masked_recon[0, 0].numpy()
        
        truth_masked_mse = np.mean((original_np - truth_masked_np) ** 2)
        print(f"  Truth-masked MSE: {truth_masked_mse:.6f}")
        
        # Explain the discrepancy
        print(f"\n💡 EXPLANATION OF DISCREPANCY:")
        print(f"  Training loss (~{loss.item():.3f}) = MSE on MASKED patches only, in patch space")
        print(f"  Visualization MSE (~{truth_masked_mse:.6f}) = MSE on full volume with VISIBLE patches intact")
        print(f"  The visualization MSE is artificially low because {(mask==0).sum().item()} patches")
        print(f"  out of {mask.numel()} total patches are original (perfect reconstruction)!")
        
        print(f"\n⚠️ VERDICT:")
        if loss.item() > 0.5:
            print(f"  Training loss of {loss.item():.3f} indicates POOR reconstruction quality")
            print(f"  The model is NOT learning effectively")
        elif loss.item() > 0.2:
            print(f"  Training loss of {loss.item():.3f} indicates MODERATE reconstruction quality")
        else:
            print(f"  Training loss of {loss.item():.3f} indicates GOOD reconstruction quality")

if __name__ == "__main__":
    debug_loss_discrepancy() 