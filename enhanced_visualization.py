#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import torch
import wandb
from pathlib import Path

def create_patch_mask_overlay(volume_shape, patch_size, mask, alpha=0.3):
    """Create a visual overlay showing which patches were masked."""
    D, H, W = volume_shape
    pd, ph, pw = patch_size
    
    # Create overlay volume
    overlay = np.zeros(volume_shape)
    
    # Calculate number of patches in each dimension
    num_patches_d = D // pd
    num_patches_h = H // ph  
    num_patches_w = W // pw
    
    # Fill overlay based on mask
    patch_idx = 0
    for d in range(num_patches_d):
        for h in range(num_patches_h):
            for w in range(num_patches_w):
                if patch_idx < len(mask):
                    # mask is 1 for masked patches, 0 for visible
                    mask_value = mask[patch_idx].item() if hasattr(mask[patch_idx], 'item') else mask[patch_idx]
                    
                    d_start, d_end = d * pd, (d + 1) * pd
                    h_start, h_end = h * ph, (h + 1) * ph
                    w_start, w_end = w * pw, (w + 1) * pw
                    
                    # 0 = visible (green), 1 = masked (red)
                    overlay[d_start:d_end, h_start:h_end, w_start:w_end] = mask_value
                    patch_idx += 1
    
    return overlay

def create_truth_masked_reconstruction(model, volumes, pred, mask):
    """Create reconstruction combining original visible patches with predicted masked patches."""
    # Patchify the original volume
    original_patches = model.patchify(volumes)  # [N, L, patch_dim]
    
    # Expand mask to match patch dimensions
    mask_expanded = mask.unsqueeze(-1).expand_as(original_patches)
    
    # Combine: use original where mask=0 (visible), predicted where mask=1 (masked)
    combined_patches = torch.where(mask_expanded == 0, original_patches, pred)
    
    # Unpatchify to get the combined volume
    combined_volume = model.unpatchify(combined_patches)
    
    return combined_volume

def create_colored_slice_visualization(original_slice, recon_slice, mask_overlay_slice, 
                                     title_prefix="", vmin=0, vmax=1):
    """Create a color-coded visualization showing visible vs masked regions."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(original_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'{title_prefix} Original')
    axes[0].axis('off')
    
    # Reconstruction
    axes[1].imshow(recon_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'{title_prefix} Reconstruction')
    axes[1].axis('off')
    
    # Color-coded overlay (Green=visible, Red=masked)
    # Create RGB image
    colored_recon = np.stack([recon_slice] * 3, axis=-1)  # Convert to RGB
    
    # Apply color coding based on mask
    visible_mask = (mask_overlay_slice == 0)  # Visible patches
    masked_mask = (mask_overlay_slice == 1)   # Masked patches
    
    # Green tint for visible regions
    colored_recon[visible_mask, 1] = np.minimum(1.0, colored_recon[visible_mask, 1] + 0.2)  # Add green
    
    # Red tint for masked regions  
    colored_recon[masked_mask, 0] = np.minimum(1.0, colored_recon[masked_mask, 0] + 0.2)  # Add red
    
    axes[2].imshow(colored_recon)
    axes[2].set_title(f'{title_prefix} Color-Coded\n(Green=Visible, Red=Masked)')
    axes[2].axis('off')
    
    # Difference map
    diff = np.abs(original_slice - recon_slice)
    im = axes[3].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
    axes[3].set_title(f'{title_prefix} |Difference|')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    return fig

def create_combined_3d_pointcloud(original_volume, recon_volume, mask_overlay, 
                                threshold=0.5, max_points=5000):
    """Create combined 3D point cloud with color coding for visible vs masked regions."""
    
    # Get points above threshold
    orig_points = np.argwhere(original_volume > threshold)
    recon_points = np.argwhere(recon_volume > threshold)
    
    if len(orig_points) == 0 or len(recon_points) == 0:
        return None, None
    
    # Subsample if too many points
    if len(orig_points) > max_points:
        indices = np.random.choice(len(orig_points), max_points, replace=False)
        orig_points = orig_points[indices]
    
    if len(recon_points) > max_points:
        indices = np.random.choice(len(recon_points), max_points, replace=False)
        recon_points = recon_points[indices]
    
    # Determine colors based on mask overlay
    orig_colors = []
    for point in orig_points:
        d, h, w = point
        if mask_overlay[d, h, w] == 0:  # Visible
            orig_colors.append([0, 1, 0])  # Green
        else:  # Masked
            orig_colors.append([1, 0, 0])  # Red
    
    recon_colors = []
    for point in recon_points:
        d, h, w = point
        if mask_overlay[d, h, w] == 0:  # Visible
            recon_colors.append([0, 0.7, 0])  # Dark green
        else:  # Masked  
            recon_colors.append([0.7, 0, 0])  # Dark red
    
    # Combine original and reconstructed points with offset
    combined_points = np.vstack([
        orig_points,
        recon_points + np.array([0, 0, original_volume.shape[2] + 10])  # Offset reconstruction
    ])
    
    combined_colors = np.vstack([orig_colors, recon_colors])
    
    return combined_points.astype(np.float32), combined_colors.astype(np.float32)

def enhanced_visualize_reconstructions(model, dataloader, device, epoch, mask_ratio, 
                                     dataset_name, num_examples=5, save_dir="enhanced_visualizations"):
    """Enhanced visualization with color-coded patches and improved 3D point clouds."""
    model.eval()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    examples_shown = 0
    individual_paths = []
    
    with torch.no_grad():
        for batch_volumes in dataloader:
            if examples_shown >= num_examples:
                break
                
            volumes_to_device = batch_volumes.to(device)
            
            # Get model predictions and mask
            loss_batch, pred_batch, mask_batch, patch_stats = model(volumes_to_device, mask_ratio=mask_ratio)
            
            # Denormalize predictions if needed
            pred_for_unpatchify = pred_batch.cpu()
            if patch_stats is not None:
                mean, var = patch_stats
                pred_for_unpatchify = pred_for_unpatchify * (var.add(1.e-6).sqrt()).cpu() + mean.cpu()
            
            # Create truth-masked reconstruction
            truth_masked_recon = create_truth_masked_reconstruction(
                model, batch_volumes, pred_for_unpatchify, mask_batch.cpu()
            )
            
            # Get numpy arrays
            original_batch_np = batch_volumes.cpu().numpy()
            reconstructed_batch_np = model.unpatchify(pred_for_unpatchify).numpy()
            truth_masked_np = truth_masked_recon.numpy()
            
            for i in range(volumes_to_device.size(0)):
                if examples_shown >= num_examples:
                    break
                    
                original_np = original_batch_np[i, 0]  # Remove channel dim
                reconstructed_np = reconstructed_batch_np[i, 0]
                truth_masked_np_i = truth_masked_np[i, 0]
                mask_np = mask_batch[i].cpu().numpy()
                
                # Create patch mask overlay
                patch_size = model.patch_size if hasattr(model, 'patch_size') else (16, 16, 16)
                mask_overlay = create_patch_mask_overlay(original_np.shape, patch_size, mask_np)
                
                # Calculate metrics
                mse_full = np.mean((original_np - reconstructed_np)**2)
                mse_truth_masked = np.mean((original_np - truth_masked_np_i)**2)
                
                # Create enhanced visualization
                fig = plt.figure(figsize=(24, 18))
                gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.3)
                
                D, H, W = original_np.shape
                mid_z, mid_y, mid_x = D//2, H//2, W//2
                
                # Row 1: Axial slices with color coding
                slice_fig = create_colored_slice_visualization(
                    original_np[mid_z], reconstructed_np[mid_z], mask_overlay[mid_z],
                    f"Axial (Z={mid_z})"
                )
                
                # Row 2: Sagittal slices  
                slice_fig2 = create_colored_slice_visualization(
                    original_np[:, :, mid_x], reconstructed_np[:, :, mid_x], mask_overlay[:, :, mid_x],
                    f"Sagittal (X={mid_x})"
                )
                
                # Row 3: Coronal slices
                slice_fig3 = create_colored_slice_visualization(
                    original_np[:, mid_y, :], reconstructed_np[:, mid_y, :], mask_overlay[:, mid_y, :],
                    f"Coronal (Y={mid_y})"
                )
                
                # Row 4: Truth-masked comparison
                axes_truth = []
                for j in range(3):
                    ax = fig.add_subplot(gs[3, j])
                    axes_truth.append(ax)
                
                axes_truth[0].imshow(truth_masked_np_i[mid_z], cmap='gray', vmin=0, vmax=1)
                axes_truth[0].set_title(f'Truth-Masked Axial\nMSE: {mse_truth_masked:.4f}')
                axes_truth[0].axis('off')
                
                axes_truth[1].imshow(truth_masked_np_i[:, :, mid_x], cmap='gray', vmin=0, vmax=1)
                axes_truth[1].set_title(f'Truth-Masked Sagittal')
                axes_truth[1].axis('off')
                
                axes_truth[2].imshow(truth_masked_np_i[:, mid_y, :], cmap='gray', vmin=0, vmax=1)
                axes_truth[2].set_title(f'Truth-Masked Coronal')
                axes_truth[2].axis('off')
                
                # Add patch grid visualization
                ax_patch = fig.add_subplot(gs[3, 3])
                patch_vis = np.zeros_like(original_np[mid_z])
                patch_vis[mask_overlay[mid_z] == 1] = 1  # Show masked patches
                ax_patch.imshow(patch_vis, cmap='Reds', alpha=0.7)
                ax_patch.imshow(original_np[mid_z], cmap='gray', alpha=0.5)
                ax_patch.set_title('Masked Patches\n(Red = Masked)')
                ax_patch.axis('off')
                
                # Add statistics
                ax_stats = fig.add_subplot(gs[3, 4:])
                stats_text = f"""
Reconstruction Statistics:
• Full Reconstruction MSE: {mse_full:.4f}
• Truth-Masked MSE: {mse_truth_masked:.4f}
• Mask Ratio: {mask_ratio*100:.1f}%
• Visible Patches: {np.sum(mask_np == 0)}/{len(mask_np)} ({np.sum(mask_np == 0)/len(mask_np)*100:.1f}%)
• Masked Patches: {np.sum(mask_np == 1)}/{len(mask_np)} ({np.sum(mask_np == 1)/len(mask_np)*100:.1f}%)

Intensity Statistics:
• Original: min={original_np.min():.3f}, max={original_np.max():.3f}, mean={original_np.mean():.3f}
• Reconstructed: min={reconstructed_np.min():.3f}, max={reconstructed_np.max():.3f}, mean={reconstructed_np.mean():.3f}
• Truth-Masked: min={truth_masked_np_i.min():.3f}, max={truth_masked_np_i.max():.3f}, mean={truth_masked_np_i.mean():.3f}
                """
                ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                            fontsize=10, verticalalignment='top', fontfamily='monospace')
                ax_stats.axis('off')
                
                fig.suptitle(f'Enhanced MAE Visualization - Epoch {epoch} - {dataset_name.capitalize()} Example {examples_shown+1}\n'
                           f'Green=Visible Patches, Red=Masked Patches', fontsize=16)
                
                # Save enhanced visualization
                enhanced_path = Path(save_dir) / f"{dataset_name}_epoch_{epoch}_example_{examples_shown+1}_enhanced.png"
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                fig.savefig(enhanced_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # Close the slice figures
                plt.close(slice_fig)
                plt.close(slice_fig2) 
                plt.close(slice_fig3)
                
                individual_paths.append(str(enhanced_path))
                
                # Create enhanced 3D point cloud for first example
                if examples_shown == 0:
                    try:
                        combined_points, combined_colors = create_combined_3d_pointcloud(
                            original_np, reconstructed_np, mask_overlay
                        )
                        
                        if combined_points is not None:
                            # Log combined point cloud
                            wandb.log({
                                f"{dataset_name}/enhanced_3d_comparison": wandb.Object3D({
                                    "type": "lidar/beta",
                                    "points": combined_points,
                                    "colors": combined_colors
                                })
                            }, step=epoch)
                            
                            # Also log separate visible and masked point clouds
                            visible_points = np.argwhere((original_np > 0.5) & (mask_overlay == 0))
                            masked_points = np.argwhere((original_np > 0.5) & (mask_overlay == 1))
                            
                            if len(visible_points) > 0:
                                wandb.log({
                                    f"{dataset_name}/visible_patches_3d": wandb.Object3D(visible_points.astype(np.float32))
                                }, step=epoch)
                            
                            if len(masked_points) > 0:
                                wandb.log({
                                    f"{dataset_name}/masked_patches_3d": wandb.Object3D(masked_points.astype(np.float32))
                                }, step=epoch)
                                
                    except Exception as e:
                        print(f"Error creating enhanced 3D visualization: {e}")
                
                examples_shown += 1
    
    return individual_paths

# Integration function to replace the existing visualize_reconstructions
def replace_visualization_in_training():
    """
    Instructions for integrating enhanced visualization:
    
    1. Replace the call to visualize_reconstructions() in test_membrane_mae.py with:
       enhanced_paths = enhanced_visualize_reconstructions(
           vis_model, vis_train_loader, device, epoch + 1, current_mask_ratio, "train", args.vis_samples
       )
       
    2. Update the wandb logging to:
       if enhanced_paths:
           log_dict["train_enhanced_visualizations"] = [
               wandb.Image(p, caption=f"Enhanced Train E{epoch+1} S{i}") 
               for i, p in enumerate(enhanced_paths)
           ]
    
    3. The enhanced visualization provides:
       - Color-coded visible (green) vs masked (red) patches
       - Truth-masked reconstruction showing perfect visible + predicted masked
       - Combined 3D point clouds with color coding
       - Detailed statistics and patch grid overlays
       - MSE comparisons between different reconstruction methods
    """
    pass 