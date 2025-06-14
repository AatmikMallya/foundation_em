import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import wandb
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg') # Explicitly set backend for headless environments
from tqdm import tqdm
import os
from pathlib import Path
import copy

# Added for potential performance improvement
torch.backends.cudnn.benchmark = True

from vit_3d import mae_vit_3d_small, mae_vit_3d_base, mae_vit_3d_large, mae_vit_3d_huge, mae_vit_3d_hemibrain_optimal, get_device
from membrane_synthetic_data import create_membrane_dataloader

# --- EMA (Exponential Moving Average) Implementation ---
class EMAModel:
    """Exponential Moving Average model for better validation performance."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self.num_updates = 0
    
    def update(self, model):
        """Update EMA model with current model parameters."""
        self.num_updates += 1
        # Adjust decay based on number of updates for better early training
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
    
    def get_model(self):
        """Get the EMA model for evaluation."""
        return self.ema_model

# --- New Learning Rate Scheduler with Warmup ---
def get_lr_scheduler_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6, initial_lr=1e-4):
    """
    Creates a learning rate scheduler with a linear warmup phase followed by cosine annealing.
    """
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Linear warmup
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        # Cosine annealing phase
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        # Ensure that the learning rate doesn't go below min_lr relative to initial_lr
        # The scheduler multiplies the optimizer's initial LR by the output of this lambda
        # So, to get an effective LR of min_lr, lambda should output min_lr / initial_lr
        decayed_lr_multiplier = cosine_decay
        min_lr_multiplier = min_lr / initial_lr
        return max(min_lr_multiplier, decayed_lr_multiplier)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --- New Progressive Mask Schedule ---
def get_progressive_mask_ratio(epoch, total_epochs,
                               initial_ratio, final_ratio,
                               warmup_phase=0.1, ramp_phase=0.3):
    """
    Calculates the mask ratio for a given epoch based on a progressive schedule
    ideal for learning both fine details and global context.

    Phase 1: Warmup with a low mask ratio to learn local features.
    Phase 2: Ramp up the difficulty to learn context.
    Phase 3: Cooldown with a high mask ratio to refine global structure.
    """
    warmup_epochs = int(total_epochs * warmup_phase)
    ramp_epochs = int(total_epochs * ramp_phase)
    
    if epoch < warmup_epochs:
        # Phase 1: Constant low mask ratio
        return initial_ratio
    elif epoch < warmup_epochs + ramp_epochs:
        # Phase 2: Linearly ramp up the mask ratio
        progress_in_ramp = (epoch - warmup_epochs) / ramp_epochs
        return initial_ratio + progress_in_ramp * (final_ratio - initial_ratio)
    else:
        # Phase 3: Constant high mask ratio
        return final_ratio

def visualize_reconstructions(model, dataloader, device, epoch, mask_ratio, dataset_name, num_examples=5, save_dir="membrane_visualizations"):
    """Visualize reconstructions, log 3D point clouds for the first example, and return paths for 2D images."""
    model.eval()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    examples_shown = 0
    all_originals_np = []
    all_reconstructed_np = []
    # For collecting 2D image paths
    individual_original_image_paths = []
    individual_recon_image_paths = []
    summary_image_path = None

    with torch.no_grad():
        for batch_volumes in dataloader:
            if examples_shown >= num_examples:
                break
            volumes_to_device = batch_volumes.to(device)
            # Model now returns: loss, predicted_volumes (patchified), mask, patch_stats
            loss_batch, pred_batch, _, patch_stats = model(volumes_to_device, mask_ratio=mask_ratio)
            
            # Denormalize predictions if norm_pix_loss is enabled
            pred_for_unpatchify = pred_batch.cpu()
            if patch_stats is not None:
                mean, var = patch_stats
                pred_for_unpatchify = pred_for_unpatchify * (var.add(1.e-6).sqrt()).cpu() + mean.cpu()
        
            reconstructed_batch_np = model.unpatchify(pred_for_unpatchify).numpy()
            original_batch_np = batch_volumes.cpu().numpy() # Keep originals on CPU

            for i in range(volumes_to_device.size(0)):
                if examples_shown >= num_examples:
                    break
                all_originals_np.append(original_batch_np[i, 0]) # Assuming (B, C, D, H, W) and C=1
                all_reconstructed_np.append(reconstructed_batch_np[i, 0])
                # all_losses.append(loss_batch.item() / volumes_to_device.size(0)) # Not used in this version
                examples_shown += 1

    if not all_originals_np:
        print(f"No examples to visualize for {dataset_name}.")
        return [], [], None

    # Create enhanced visualization with slice grids and orthogonal views for each example
    for i in range(examples_shown):
        original_np = all_originals_np[i]
        reconstructed_np = all_reconstructed_np[i]
        mse = np.mean((original_np - reconstructed_np)**2)
        
        # --- Save individual original image ---
        fig_orig = plt.figure(figsize=(20, 12))
        gs_orig = fig_orig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        D, H, W = original_np.shape
        slice_indices = np.linspace(D//6, 5*D//6, 9, dtype=int)
        for j, slice_idx in enumerate(slice_indices):
            row, col = j // 3, j % 3
            ax = fig_orig.add_subplot(gs_orig[row, col])
            ax.imshow(original_np[slice_idx], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Orig Z={slice_idx}', fontsize=8)
            ax.axis('off')
        mid_z, mid_y, mid_x = D//2, H//2, W//2
        ax_orig_axial = fig_orig.add_subplot(gs_orig[0, 3])
        ax_orig_axial.imshow(original_np[mid_z], cmap='gray', vmin=0, vmax=1); ax_orig_axial.set_title(f'Original Axial (Z={mid_z})'); ax_orig_axial.axis('off')
        ax_orig_sagittal = fig_orig.add_subplot(gs_orig[1, 3])
        ax_orig_sagittal.imshow(original_np[:, :, mid_x], cmap='gray', vmin=0, vmax=1); ax_orig_sagittal.set_title(f'Original Sagittal (X={mid_x})'); ax_orig_sagittal.axis('off')
        ax_orig_coronal = fig_orig.add_subplot(gs_orig[2, 3])
        ax_orig_coronal.imshow(original_np[:, mid_y, :], cmap='gray', vmin=0, vmax=1); ax_orig_coronal.set_title(f'Original Coronal (Y={mid_y})'); ax_orig_coronal.axis('off')
        fig_orig.suptitle(f'Epoch {epoch} - {dataset_name.capitalize()} Example {i+1} - ORIGINAL\nMask {mask_ratio*100:.0f}%', fontsize=16)
        
        current_original_path = Path(save_dir) / f"{dataset_name}_epoch_{epoch}_example_{i+1}_original_slices.png"
        fig_orig.tight_layout()
        fig_orig.savefig(current_original_path, dpi=100, bbox_inches='tight') # Lower dpi for faster save
        plt.close(fig_orig)
        individual_original_image_paths.append(str(current_original_path))

        # --- Save individual reconstructed image ---
        fig_recon = plt.figure(figsize=(20, 12))
        gs_recon = fig_recon.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        for j, slice_idx in enumerate(slice_indices):
            row, col = j // 3, j % 3
            ax = fig_recon.add_subplot(gs_recon[row, col])
            ax.imshow(reconstructed_np[slice_idx], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Recon Z={slice_idx}', fontsize=8)
            ax.axis('off')
        ax_recon_axial = fig_recon.add_subplot(gs_recon[0, 3])
        ax_recon_axial.imshow(reconstructed_np[mid_z], cmap='gray', vmin=0, vmax=1); ax_recon_axial.set_title(f'Reconstructed Axial (Z={mid_z})'); ax_recon_axial.axis('off')
        ax_recon_sagittal = fig_recon.add_subplot(gs_recon[1, 3])
        ax_recon_sagittal.imshow(reconstructed_np[:, :, mid_x], cmap='gray', vmin=0, vmax=1); ax_recon_sagittal.set_title(f'Reconstructed Sagittal (X={mid_x})'); ax_recon_sagittal.axis('off')
        ax_recon_coronal = fig_recon.add_subplot(gs_recon[2, 3])
        ax_recon_coronal.imshow(reconstructed_np[:, mid_y, :], cmap='gray', vmin=0, vmax=1); ax_recon_coronal.set_title(f'Reconstructed Coronal (Y={mid_y})'); ax_recon_coronal.axis('off')
        fig_recon.suptitle(f'Epoch {epoch} - {dataset_name.capitalize()} Example {i+1} - RECONSTRUCTION\nMask {mask_ratio*100:.0f}% - MSE: {mse:.4f}', fontsize=16)
        
        current_recon_path = Path(save_dir) / f"{dataset_name}_epoch_{epoch}_example_{i+1}_reconstruction_slices.png"
        fig_recon.tight_layout()
        fig_recon.savefig(current_recon_path, dpi=100, bbox_inches='tight') # Lower dpi
        plt.close(fig_recon)
        individual_recon_image_paths.append(str(current_recon_path))

        # --- Log 3D Point Cloud for the FIRST example only ---
        if i == 0:
            try:
                points_orig = np.argwhere(original_np > 0.5).astype(np.float32)
                if points_orig.shape[0] > 0:
                    wandb.log({f"{dataset_name}/example_0_original_PointCould": wandb.Object3D(points_orig)}, step=epoch)
                else:
                    print(f"WDB: No points in original for 3D cloud (epoch {epoch}, {dataset_name})")
                
                points_recon = np.argwhere(reconstructed_np > 0.5).astype(np.float32)
                if points_recon.shape[0] > 0:
                    wandb.log({f"{dataset_name}/example_0_reconstructed_PointCloud": wandb.Object3D(points_recon)}, step=epoch)
                else:
                    print(f"WDB: No points in reconstruction for 3D cloud (epoch {epoch}, {dataset_name})")
            except Exception as e_3d:
                print(f"Error logging 3D point cloud for {dataset_name} example {i}, epoch {epoch}: {e_3d}")
    
    # Create a summary comparison figure (using the first example)
    if examples_shown > 0:
        avg_mse_all_examples = np.mean([np.mean((all_originals_np[j] - all_reconstructed_np[j])**2) for j in range(examples_shown)])
        
        fig_summary, axes_summary = plt.subplots(2, 3, figsize=(15, 10))
        first_orig_np = all_originals_np[0]
        first_recon_np = all_reconstructed_np[0]
        D_s, H_s, W_s = first_orig_np.shape
        mid_z_s, mid_y_s, mid_x_s = D_s//2, H_s//2, W_s//2
        
        axes_summary[0, 0].imshow(first_orig_np[mid_z_s], cmap='gray', vmin=0, vmax=1); axes_summary[0, 0].set_title('Original Axial'); axes_summary[0, 0].axis('off')
        axes_summary[0, 1].imshow(first_orig_np[:, :, mid_x_s], cmap='gray', vmin=0, vmax=1); axes_summary[0, 1].set_title('Original Sagittal'); axes_summary[0, 1].axis('off')
        axes_summary[0, 2].imshow(first_orig_np[:, mid_y_s, :], cmap='gray', vmin=0, vmax=1); axes_summary[0, 2].set_title('Original Coronal'); axes_summary[0, 2].axis('off')
        
        axes_summary[1, 0].imshow(first_recon_np[mid_z_s], cmap='gray', vmin=0, vmax=1); axes_summary[1, 0].set_title('Reconstructed Axial'); axes_summary[1, 0].axis('off')
        axes_summary[1, 1].imshow(first_recon_np[:, :, mid_x_s], cmap='gray', vmin=0, vmax=1); axes_summary[1, 1].set_title('Reconstructed Sagittal'); axes_summary[1, 1].axis('off')
        axes_summary[1, 2].imshow(first_recon_np[:, mid_y_s, :], cmap='gray', vmin=0, vmax=1); axes_summary[1, 2].set_title('Reconstructed Coronal'); axes_summary[1, 2].axis('off')
        
        fig_summary.suptitle(f'Epoch {epoch} - {dataset_name.capitalize()} Summary (Example 0)\nMask {mask_ratio*100:.0f}% - Avg MSE over {examples_shown} eg: {avg_mse_all_examples:.4f}', fontsize=14)
        fig_summary.tight_layout()
        
        summary_image_path_obj = Path(save_dir) / f"{dataset_name}_epoch_{epoch}_summary_comparison.png"
        fig_summary.savefig(summary_image_path_obj, dpi=100, bbox_inches='tight') # Lower dpi
        plt.close(fig_summary)
        summary_image_path = str(summary_image_path_obj)
        print(f"Enhanced 2D/3D visualizations generated for {dataset_name}, epoch {epoch}. Summary: {summary_image_path}")

    return individual_original_image_paths, individual_recon_image_paths, summary_image_path

def main(args):
    device = get_device()
    print(f"Using device: {device}")
    
    # Enable TensorFloat32 for better performance on Ampere GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print("Enabled TensorFloat32 for improved performance")
    
    # Initialize training profiler
    from training_profiler import TrainingProfiler
    profiler = TrainingProfiler(log_interval=5, memory_tracking=True, detailed_timing=True)

    # Parameters are now controlled via argparse defaults and command-line arguments
    # for better flexibility and experiment tracking.

    # Ensure patch_size is a tuple
    patch_size = tuple(map(int, args.patch_size.split(',')))
    if len(patch_size) == 1:
        patch_size = (patch_size[0], patch_size[0], patch_size[0])
    elif len(patch_size) != 3:
        raise ValueError("patch_size must be a single integer or three integers separated by commas.")
    print(f"Using 3D patch size: {patch_size}")

    # WandB
    wandb_project = args.project_name if args.project_name else "mae-3d-membranes"
    run_name = args.run_name
    if args.overfit_test:
        run_name += "_overfit_test"
    wandb.init(project=wandb_project, name=run_name, config=args)
    
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    if args.overfit_test:
        wandb.define_metric("overfit_iteration")
        wandb.define_metric("overfit_loss", step_metric="overfit_iteration")
        wandb.define_metric("overfit_reconstruction", step_metric="overfit_iteration")

    # Model (same for both modes)
    print(f"Selected model architecture: {args.model_arch}")
    if args.model_arch == "small":
        model_fn = mae_vit_3d_small
    elif args.model_arch == "base":
        model_fn = mae_vit_3d_base
    elif args.model_arch == "large":
        model_fn = mae_vit_3d_large
    elif args.model_arch == "huge":
        model_fn = mae_vit_3d_huge
    elif args.model_arch == "hemibrain_optimal":
        model_fn = mae_vit_3d_hemibrain_optimal
    else:
        raise ValueError(f"Unsupported model_arch: {args.model_arch}. Choose from available options.")

    model = model_fn(
        volume_size=(args.img_size, args.img_size, args.img_size),
        patch_size=patch_size,
        norm_pix_loss=args.norm_pix_loss
    ).to(device)

    # print the image size
    print(f"Image size: {args.img_size}")

    # Compile model for better performance (PyTorch 2.0+)
    # No fallback: if compilation fails, the job should fail.
    print("Compiling model with torch.compile for improved performance...")
    model = torch.compile(model, mode='default')
    print("Model compilation successful.")
    
    wandb.watch(model, log_freq=100)
    # Optimizer now uses learning rate from args
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Initialize EMA model
    if not args.overfit_test:
        ema_model = EMAModel(model, decay=args.ema_decay)
        print(f"Initialized EMA model with decay={args.ema_decay}")
    else:
        ema_model = None
    
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("Using Automatic Mixed Precision (AMP) with GradScaler.")
    else:
        print("AMP is DISABLED. Training with full precision (float32).")

    # --- Apply new LR Scheduler with Warmup ---
    scheduler = None # Initialize scheduler to None
    if not args.overfit_test:
        # Use configurable parameters from args
        print(f"Using LR scheduler: {args.warmup_epochs}-epoch warmup then cosine annealing over {args.epochs} epochs from {args.learning_rate} to {args.min_lr}.")
        scheduler = get_lr_scheduler_with_warmup(optimizer, 
                                                 warmup_epochs=args.warmup_epochs, 
                                                 total_epochs=args.epochs, 
                                                 min_lr=args.min_lr, 
                                                 initial_lr=args.learning_rate)

    if args.overfit_test:
        print("--- RUNNING OVERFIT TEST ON A SINGLE SAMPLE ---")
        # Load a single sample
        overfit_dataset = create_membrane_dataloader(
            batch_size=1, num_samples=1, volume_size=(args.img_size, args.img_size, args.img_size),
            num_gaussians_range=args.num_spheres_range, gaussian_sigma_range=args.radius_range,
            noise_level=args.noise_level, membrane_band_width=args.membrane_band_width,
            num_additional_spheres_range=args.num_added_spheres,
            additional_sphere_radius_range=args.added_sphere_radii,
            num_workers=0, shuffle=False, seed=42, # Fixed seed for consistency
        )
        single_volume_loader = overfit_dataset # Dataloader returns the batch directly
        # Get the single volume once, it will be reused
        single_volume = next(iter(single_volume_loader)).to(device)
        if single_volume.ndim == 4: single_volume = single_volume.unsqueeze(1)

        num_overfit_iterations = 2000 # Increased iterations
        overfit_mask_ratio = 0.50 # Explicitly 50% masking
        vis_interval_overfit = 100 # Adjusted for more iterations
        print(f"Training on one sample for {num_overfit_iterations} iterations with mask ratio {overfit_mask_ratio:.2f}")

        model.train()
        for iteration in tqdm(range(num_overfit_iterations), desc="Overfitting to one sample"):
            optimizer.zero_grad()
            loss, _, _, _ = model(single_volume, mask_ratio=overfit_mask_ratio)
            if torch.isnan(loss):
                print(f"NaN loss at iteration {iteration+1}. Stopping overfit test.")
                break
            loss.backward()
            optimizer.step()
            
            wandb.log({"overfit_loss": loss.item(), "overfit_iteration": iteration + 1})

            if (iteration + 1) % vis_interval_overfit == 0 or iteration == num_overfit_iterations - 1:
                print(f"  Visualizing overfit reconstruction at iteration {iteration + 1}...")
                try:
                    # Create a temporary dataloader with the single sample for visualization function
                    temp_vis_loader = [(single_volume.cpu())] # visualize_reconstructions expects an iterable of batches
                    original_paths, recon_paths, summary_path = visualize_reconstructions(model, temp_vis_loader, device, iteration + 1, overfit_mask_ratio, "overfit_sample", num_examples=1)
                    if summary_path:
                        wandb.log({"overfit_reconstruction": wandb.Image(summary_path), "overfit_iteration": iteration + 1})
                except Exception as e:
                    print(f"  Overfit visualization failed: {e}")
        print("--- OVERFIT TEST COMPLETE ---")

    else: # Regular training mode
        # Dataset and DataLoader (Original logic)
        print("Loading training dataset...")
        train_loader = create_membrane_dataloader(
            batch_size=args.batch_size, num_samples=args.train_samples, volume_size=(args.img_size, args.img_size, args.img_size),
            num_gaussians_range=args.num_spheres_range, gaussian_sigma_range=args.radius_range,
            noise_level=args.noise_level, membrane_band_width=args.membrane_band_width,
            num_additional_spheres_range=args.num_added_spheres,
            additional_sphere_radius_range=args.added_sphere_radii,
            num_workers=args.num_workers, shuffle=True, seed=0,
        )
        print(f"Training dataset loaded with {args.train_samples} samples.")

        print("Loading validation dataset...")
        val_loader = create_membrane_dataloader(
            batch_size=args.batch_size, num_samples=args.val_samples, volume_size=(args.img_size, args.img_size, args.img_size),
            num_gaussians_range=args.num_spheres_range, gaussian_sigma_range=args.radius_range,
            noise_level=args.noise_level, membrane_band_width=args.membrane_band_width,
            num_additional_spheres_range=args.num_added_spheres,
            additional_sphere_radius_range=args.added_sphere_radii,
            num_workers=args.num_workers, shuffle=False, seed=args.train_samples,
            drop_last=False  # Don't drop validation batches
        )
        print(f"Validation dataset loaded with {args.val_samples} samples.")

        vis_train_loader = create_membrane_dataloader(
            batch_size=1, num_samples=min(args.vis_samples, args.train_samples), volume_size=(args.img_size, args.img_size, args.img_size),
            num_gaussians_range=args.num_spheres_range, gaussian_sigma_range=args.radius_range,
            noise_level=args.noise_level, membrane_band_width=args.membrane_band_width,
            num_additional_spheres_range=args.num_added_spheres,
            additional_sphere_radius_range=args.added_sphere_radii,
            num_workers=args.num_workers, shuffle=False, seed=0,
            drop_last=False  # Don't drop visualization batches
        )
        vis_val_loader = create_membrane_dataloader(
            batch_size=1, num_samples=min(args.vis_samples, args.val_samples), volume_size=(args.img_size, args.img_size, args.img_size),
            num_gaussians_range=args.num_spheres_range, gaussian_sigma_range=args.radius_range,
            noise_level=args.noise_level, membrane_band_width=args.membrane_band_width,
            num_additional_spheres_range=args.num_added_spheres,
            additional_sphere_radius_range=args.added_sphere_radii,
            num_workers=args.num_workers, shuffle=False, seed=args.train_samples,
            drop_last=False  # Don't drop validation batches for visualization
        )
        
        # print the initial and final masking ratio
        print(f"Initial masking ratio: {args.initial_masking_ratio}")
        print(f"Final masking ratio: {args.final_masking_ratio}")
        
        print(f"Starting training on membrane data for {args.epochs} epochs...")
        for epoch in range(args.epochs):
            # Get current mask ratio from the new schedule
            current_mask_ratio = get_progressive_mask_ratio(epoch + 1, args.epochs, args.initial_masking_ratio, args.final_masking_ratio) # epoch + 1 for 1-indexed
            
            # --- On-the-fly data generation ---
            with profiler.profile_section("data_epoch_setup"):
                train_loader.dataset.set_epoch(epoch)
                vis_train_loader.dataset.set_epoch(epoch)
            
            model.train()
            epoch_loss = 0
            first_batch_for_logging = None

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training] (Mask {current_mask_ratio*100:.0f}%)", unit="batch")
            
            for batch_idx, volumes in enumerate(progress_bar):
                profiler.start_batch_timing()
                if batch_idx == 0:
                    first_batch_for_logging = volumes.detach().cpu().clone()

                with profiler.profile_section("data_transfer"):
                    volumes = volumes.to(device)
                    if volumes.ndim == 4: volumes = volumes.unsqueeze(1)

                with profiler.profile_section("optimizer_zero_grad"):
                    optimizer.zero_grad(set_to_none=True)
                
                # Forward and backward pass
                if args.use_amp and scaler:
                    with profiler.profile_section("forward_pass_amp"):
                        with torch.cuda.amp.autocast():
                            loss, _, _, _ = model(volumes, mask_ratio=current_mask_ratio)
                    if torch.isnan(loss):
                        print(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx} (AMP). Skipping batch.")
                        continue
                    with profiler.profile_section("backward_pass_amp"):
                        scaler.scale(loss).backward()
                    with profiler.profile_section("gradient_clipping"):
                        if args.grad_clip_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    with profiler.profile_section("optimizer_step_amp"):
                        scaler.step(optimizer)
                        scaler.update()
                else: 
                    with profiler.profile_section("forward_pass"):
                        loss, _, _, _ = model(volumes, mask_ratio=current_mask_ratio)
                    if torch.isnan(loss):
                        print(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                        continue
                    with profiler.profile_section("backward_pass"):
                        loss.backward()
                    with profiler.profile_section("gradient_clipping"):
                        if args.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    with profiler.profile_section("optimizer_step"):
                        optimizer.step()
                
                with profiler.profile_section("ema_update"):
                    if ema_model is not None:
                        ema_model.update(model)
            
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'], mask=f"{current_mask_ratio:.2f}")
                profiler.end_batch_timing(batch_idx)

            avg_train_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
            print(f"Epoch {epoch+1}/{args.epochs} (Mask {current_mask_ratio*100:.0f}%) - Training Loss: {avg_train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log_data_characteristic = {}
            if first_batch_for_logging is not None and first_batch_for_logging.nelement() > 0:
                sample_sum = first_batch_for_logging[0].sum().item()
                log_data_characteristic = {"epoch_first_train_batch_sample_sum": sample_sum}
                if epoch < 5 or (epoch + 1) % 100 == 0 :
                    print(f"Epoch {epoch+1} - First train batch, first sample sum: {sample_sum:.4f}")

            # Validation
            with profiler.profile_section("validation_epoch"):
                eval_model = ema_model.get_model() if ema_model else model
                eval_model.eval()
                val_loss = 0
                progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation] (Mask {current_mask_ratio*100:.0f}%)", unit="batch")
                with torch.no_grad():
                    for volumes_val in progress_bar_val:
                        with profiler.profile_section("validation_data_transfer"):
                            volumes_val = volumes_val.to(device)
                            if volumes_val.ndim == 4: volumes_val = volumes_val.unsqueeze(1)
                        with profiler.profile_section("validation_forward"):
                            with torch.cuda.amp.autocast(enabled=args.use_amp):
                                loss_val, _, _, _ = eval_model(volumes_val, mask_ratio=current_mask_ratio)
                        if torch.isnan(loss_val):
                            print(f"NaN validation loss detected at epoch {epoch+1}. Skipping batch.")
                            continue
                        val_loss += loss_val.item()
                        progress_bar_val.set_postfix(loss=loss_val.item())
        
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            print(f"Epoch {epoch+1}/{args.epochs} (Mask {current_mask_ratio*100:.0f}%) - Validation Loss (EMA): {avg_val_loss:.4f}")

            # CRITICAL FIX: Reset model to train mode if we modified it during validation
            if ema_model is None:
                model.train()  # Reset to train mode if we used the main model for validation

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log_dict = {
                "epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                "mask_ratio": current_mask_ratio, "learning_rate": optimizer.param_groups[0]['lr'],
                **log_data_characteristic
            }

            # Visualization
            if epoch == 0 or (epoch + 1) % args.vis_interval == 0 or epoch == args.epochs - 1:
                print(f"  Generating reconstructions for epoch {epoch + 1}...")
                try:
                    with profiler.profile_section("visualization_generation"):
                        from enhanced_visualization import enhanced_visualize_reconstructions
                        vis_model = ema_model.get_model() if ema_model else model
                        
                        train_enhanced_paths = enhanced_visualize_reconstructions(vis_model, vis_train_loader, device, epoch + 1, current_mask_ratio, "train", args.vis_samples)
                        if train_enhanced_paths:
                            log_dict["train_enhanced_visualizations"] = [wandb.Image(p, caption=f"Enhanced Train E{epoch+1} S{i}") for i, p in enumerate(train_enhanced_paths)]

                        val_enhanced_paths = enhanced_visualize_reconstructions(vis_model, vis_val_loader, device, epoch + 1, current_mask_ratio, "val", args.vis_samples)
                        if val_enhanced_paths:
                            log_dict["val_enhanced_visualizations"] = [wandb.Image(p, caption=f"Enhanced Val E{epoch+1} S{i}") for i, p in enumerate(val_enhanced_paths)]
                except Exception as e:
                    print(f"  Visualization failed for epoch {epoch + 1}: {e}")

            wandb.log(log_dict)
            profiler.log_epoch_summary(epoch + 1)

            if scheduler:
                scheduler.step()

            # Checkpointing
            if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
                checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"mae_membrane_epoch_{epoch+1}.pth")
                
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'args': args
                }
                if ema_model is not None:
                    checkpoint_data['ema_model_state_dict'] = ema_model.get_model().state_dict()
                    checkpoint_data['ema_num_updates'] = ema_model.num_updates
                
                torch.save(checkpoint_data, checkpoint_path)
                wandb.save(checkpoint_path)
                print(f"Saved checkpoint (including EMA) to {checkpoint_path}")
        
        # Cleanup after regular training loop
        profiler.cleanup()
    
    wandb.finish()
    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train 3D Masked Autoencoder on Synthetic Membrane Data")
    
    # Dataset parameters
    parser.add_argument('--img_size', type=int, default=64, help='Size of the 3D images (cubic, e.g., 64 for 64x64x64)')
    # MembraneSyntheticDataset specific
    parser.add_argument('--num_spheres_range', type=str, default="12,15", help='Range for number of spheres (min,max)')
    parser.add_argument('--radius_range', type=str, default="12,18", help='Range for sphere radii (min,max)')
    parser.add_argument('--noise_level', type=float, default=0.01, help='Noise level for the synthetic data (improved from 0.001)')
    parser.add_argument('--membrane_band_width', type=float, default=0.1, help='Membrane band width for synthetic data')
    parser.add_argument('--train_samples', type=int, default=4096, help='Number of samples in training set')
    parser.add_argument('--val_samples', type=int, default=512, help='Number of samples in validation set')
    parser.add_argument('--num_added_spheres', type=str, default="4,4", help='Range for number of added spheres (min,max)')
    parser.add_argument('--added_sphere_radii', type=str, default="5.0,5.0", help='Range for added sphere radii (min,max)')

    # MAE & ViT Architecture parameters
    parser.add_argument('--patch_size', type=str, default="16", help="Patch size for ViT (single int for cubic, or D,H,W).")
    # Note: mae_vit_3d_small() uses fixed architecture (embed_dim=384, depth=8, etc.)
    # The following parameters are kept for compatibility but not used by mae_vit_3d_small()
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension for ViT encoder')

    parser.add_argument('--decoder_dim', type=int, default=512, help='Dimension of MAE decoder')
    parser.add_argument('--decoder_depth', type=int, default=8, help='Depth of MAE decoder')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=2500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training (reduced default for memory efficiency)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate for the scheduler')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='Minimum learning rate for the cosine annealing scheduler (improved from 1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=100, help='Number of warmup epochs for the learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='Weight decay for AdamW optimizer (improved from 0.05)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision training')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate for model averaging')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use per-patch normalized pixels as loss target.')

    # Initial and final masking ratio
    parser.add_argument('--initial_masking_ratio', type=float, default=0.40, help='Initial masking ratio')
    parser.add_argument('--final_masking_ratio', type=float, default=0.40, help='Final masking ratio')

    # Logging and Saving
    parser.add_argument('--run_name', type=str, default='mae_membrane_run_v4_ema', help='Name of the W&B run')
    parser.add_argument('--project_name', type=str, default='mae-3d-membranes', help='Name of the W&B project')
    parser.add_argument('--vis_interval', type=int, default=100, help='Epoch interval for visualizing reconstructions')
    parser.add_argument('--vis_samples', type=int, default=2, help='Number of samples to visualize')
    parser.add_argument('--save_interval', type=int, default=250, help='Epoch interval for saving model checkpoints')
    parser.add_argument('--overfit_test', action='store_true', help='Run an overfitting test on a single sample.')
    parser.add_argument('--model_arch', type=str, default="small", choices=["small", "base", "large", "huge", "hemibrain_optimal"], help="Model architecture to use.")

    args = parser.parse_args()
    
    # Process range arguments
    args.num_spheres_range = tuple(map(int, args.num_spheres_range.split(',')))
    args.radius_range = tuple(map(int, args.radius_range.split(',')))
    args.num_added_spheres = tuple(map(int, args.num_added_spheres.split(',')))
    args.added_sphere_radii = tuple(map(float, args.added_sphere_radii.split(',')))

    main(args) 