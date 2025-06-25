#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAE Feature Inspector
====================
Loads a trained SAE checkpoint and visualizes what each latent responds to
in the 3D EM data. Creates overlays showing where latents fire on actual
membrane/organelle structures.

Usage:
    python3 inspect_sae.py --sae_checkpoint checkpoints/sae/sae_layer6_lat8_mask50.pt
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Project imports
from vit_3d import mae_vit_3d_base_conv, get_device
from sae_train import extract_patch_tokens
from vol_train import TarShardDataset, CUDAPrefetcher
from torch.utils.data import DataLoader

def load_sae_checkpoint(ckpt_path):
    """Load SAE checkpoint and extract key parameters."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    return {
        'weight': ckpt['sae_weight'],
        'decoder_bias': ckpt.get('decoder_bias', None),  # Handle old checkpoints without bias
        'input_dim': ckpt['input_dim'], 
        'latent_dim': ckpt['latent_dim'],
        'layer': ckpt['layer'],
        'l1_coeff': ckpt['l1_coeff'],
        'config': ckpt['config']
    }

def load_mae_model(config, device):
    """Load the pretrained MAE model used for feature extraction."""
    # Determine architecture
    if config['model_arch'] == 'base_conv':
        mae = mae_vit_3d_base_conv(
            volume_size=(config['img_size'],) * 3,
            patch_size=config['patch_size'],
            mask_ratio=0.0
        ).to(device)
    else:
        raise ValueError(f"Unsupported model arch: {config['model_arch']}")
    
    # Load checkpoint weights
    mae_ckpt = torch.load(config['checkpoint'], map_location="cpu")
    state_dict = mae_ckpt["model_state_dict"]
    
    # Remove _orig_mod. prefix from torch.compile
    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            clean_key = key[len('_orig_mod.'):]
            clean_state_dict[clean_key] = value
        else:
            clean_state_dict[key] = value
    
    mae.load_state_dict(clean_state_dict, strict=False)
    mae.eval()
    
    # Freeze all parameters
    for p in mae.parameters():
        p.requires_grad = False
        
    return mae

def get_validation_data(shard_dir, img_size, batch_size=4, device='cuda'):
    """Load a batch of validation volumes."""
    shard_path = Path(shard_dir)
    shards = sorted(shard_path.glob("shard*.tar"))
    
    if not shards:
        raise FileNotFoundError(f"No shard*.tar files found in {shard_dir}")
    
    print(f"Found {len(shards)} shards: {[s.name for s in shards[:3]]}...")
    
    # Use first shard for validation
    val_ds = TarShardDataset([shards[0]], img_size, shuffle=False)
    print(f"Dataset: {len(shards)} shards, shuffle=False")
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        num_workers=2,  # Reduce workers to avoid CUDA issues
        pin_memory=False,  # Disable pin_memory to avoid CUDA errors
        drop_last=False
    )
    
    print(f"DataLoader created with batch_size={batch_size}")
    
    # Try to get data without CUDA prefetcher first
    try:
        # Get one batch directly from loader
        for volumes in val_loader:
            # Move to device manually
            volumes = volumes.to(device, non_blocking=True)
            print(f"Loaded batch shape: {volumes.shape}")
            return volumes
        
        # If we get here, the loader was empty
        raise RuntimeError("DataLoader is empty - no data found in shards")
        
    except Exception as e:
        print(f"Error with DataLoader: {e}")
        print("Trying fallback method...")
        
        # Fallback: Load data on CPU first
        val_loader_cpu = DataLoader(
            val_ds, 
            batch_size=batch_size, 
            num_workers=0,  # No multiprocessing
            pin_memory=False,
            drop_last=False
        )
        
        for volumes in val_loader_cpu:
            volumes = volumes.to(device)
            print(f"Loaded batch shape (fallback): {volumes.shape}")
            return volumes
            
        raise RuntimeError("Both DataLoader methods failed")

def compute_sae_activations(mae, volumes, sae_weight, decoder_bias, layer_idx, device):
    """Extract tokens and compute SAE activations."""
    # Get patch tokens from specified layer
    with torch.no_grad():
        tokens = extract_patch_tokens(mae, volumes, layer_idx)  # (N_tokens, C)
        
        # Apply SAE with decoder bias (if available)
        if decoder_bias is not None:
            # Follow Anthropic method: subtract decoder bias before encoding
            tokens_centered = tokens - decoder_bias.to(device)
            z = torch.relu(tokens_centered @ sae_weight.T)  # (N_tokens, latent_dim)
        else:
            # Old method for backward compatibility
            z = torch.relu(tokens @ sae_weight.T)  # (N_tokens, latent_dim)
        
    return tokens, z

def patch_coords_to_voxel_heat(activations, patch_grid_shape, patch_size):
    """Convert patch activations to full-resolution 3D heatmap."""
    # Reshape patch activations to grid
    grid_d, grid_h, grid_w = patch_grid_shape
    heat_grid = activations.view(grid_d, grid_h, grid_w)
    
    # Upsample to voxel resolution using nearest neighbor (kron product)
    heat_3d = np.kron(heat_grid.cpu().numpy(), np.ones((patch_size, patch_size, patch_size)))
    
    # Normalize to [0, 1]
    heat_min, heat_max = heat_3d.min(), heat_3d.max()
    if heat_max > heat_min:
        heat_3d = (heat_3d - heat_min) / (heat_max - heat_min)
    
    return heat_3d

def visualize_latent_on_volume(volume, heatmap, latent_id, save_dir=None):
    """Create overlay visualization of latent activation on raw EM volume."""
    # Convert to numpy
    if torch.is_tensor(volume):
        volume = volume.cpu().numpy()
    
    # Normalize volume for display
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min())
    
    # Create figure with multiple slice views
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Get dimensions
    d, h, w = volume.shape
    
    # Show orthogonal slices through center
    slices = [
        (vol_norm[d//2, :, :], heatmap[d//2, :, :], 'Z-slice (sagittal)'),
        (vol_norm[:, h//2, :], heatmap[:, h//2, :], 'Y-slice (coronal)'), 
        (vol_norm[:, :, w//2], heatmap[:, :, w//2], 'X-slice (axial)')
    ]
    
    for i, (vol_slice, heat_slice, title) in enumerate(slices):
        # Raw volume
        axes[0, i].imshow(vol_slice, cmap='gray')
        axes[0, i].set_title(f'{title}\nRaw EM')
        axes[0, i].axis('off')
        
        # Overlay heatmap
        axes[1, i].imshow(vol_slice, cmap='gray')
        axes[1, i].imshow(heat_slice, cmap='hot', alpha=0.6, vmin=0, vmax=1)
        axes[1, i].set_title(f'{title}\nLatent {latent_id} Overlay')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Latent {latent_id} Activation Pattern', fontsize=14)
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / f'latent_{latent_id:04d}_overlay.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None

def analyze_latent_statistics(activations):
    """Compute statistics about latent activations."""
    with torch.no_grad():
        # Sparsity (fraction of zero activations)
        sparsity = (activations == 0).float().mean(0)  # per latent
        frac_active = 1 - sparsity
        
        # Total activation mass per latent
        total_mass = activations.sum(0)
        
        # Average activation when active
        avg_when_active = []
        for i in range(activations.shape[1]):
            active_mask = activations[:, i] > 0
            if active_mask.sum() > 0:
                avg_when_active.append(activations[active_mask, i].mean().item())
            else:
                avg_when_active.append(0.0)
        
        return {
            'sparsity': sparsity.cpu().numpy(),
            'frac_active': frac_active.cpu().numpy(), 
            'total_mass': total_mass.cpu().numpy(),
            'avg_when_active': np.array(avg_when_active)
        }

def create_summary_plots(stats, save_dir=None):
    """Create summary plots of SAE statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Sparsity histogram
    axes[0, 0].hist(stats['frac_active'] * 100, bins=50, alpha=0.7)
    axes[0, 0].axvline(2, color='red', linestyle='--', label='Target 2%')
    axes[0, 0].axvline(6, color='red', linestyle='--', label='Target 6%')
    axes[0, 0].set_xlabel('% Active (when latent fires)')
    axes[0, 0].set_ylabel('Number of Latents')
    axes[0, 0].set_title('Activation Sparsity Distribution')
    axes[0, 0].legend()
    
    # Total activation mass
    axes[0, 1].hist(np.log10(stats['total_mass'] + 1e-6), bins=50, alpha=0.7)
    axes[0, 1].set_xlabel('Log10(Total Activation Mass)')
    axes[0, 1].set_ylabel('Number of Latents')
    axes[0, 1].set_title('Total Activation Distribution')
    
    # Average activation when active
    axes[1, 0].hist(stats['avg_when_active'], bins=50, alpha=0.7)
    axes[1, 0].set_xlabel('Average Activation (when > 0)')
    axes[1, 0].set_ylabel('Number of Latents')
    axes[1, 0].set_title('Activation Magnitude Distribution')
    
    # Sparsity vs activation strength
    axes[1, 1].scatter(stats['frac_active'] * 100, stats['avg_when_active'], alpha=0.6, s=10)
    axes[1, 1].set_xlabel('% Active')
    axes[1, 1].set_ylabel('Average Activation (when > 0)')
    axes[1, 1].set_title('Sparsity vs Activation Strength')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'sae_summary_stats.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None

def main():
    parser = argparse.ArgumentParser(description='Inspect SAE learned features')
    parser.add_argument('--sae_checkpoint', required=True, help='Path to SAE checkpoint')
    parser.add_argument('--shard_dir', default='/gpfs/radev/home/am3833/scratch/volumes_96', 
                       help='Directory containing validation shards')
    parser.add_argument('--output_dir', default='sae_inspection', help='Output directory for visualizations')
    parser.add_argument('--num_latents', type=int, default=20, help='Number of top latents to visualize')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of volumes to analyze')
    
    args = parser.parse_args()
    
    # Setup device with better error handling
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize CUDA if available
    if device.type == 'cuda':
        try:
            torch.cuda.init()
            torch.cuda.empty_cache()
            print(f"CUDA initialized successfully")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
        except Exception as e:
            print(f"CUDA initialization error: {e}")
            print("Falling back to CPU")
            device = torch.device('cpu')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading SAE checkpoint from {args.sae_checkpoint}")
    sae_data = load_sae_checkpoint(args.sae_checkpoint)
    
    print(f"SAE info: {sae_data['latent_dim']} latents, layer {sae_data['layer']}")
    print(f"L1 coefficient: {sae_data['l1_coeff']}")
    
    # Load MAE model
    print("Loading MAE model...")
    mae = load_mae_model(sae_data['config'], device)
    
    # Move SAE weights to device
    sae_weight = sae_data['weight'].to(device)
    decoder_bias = sae_data['decoder_bias']
    if decoder_bias is not None:
        decoder_bias = decoder_bias.to(device)
    
    # Load validation data
    print(f"Loading validation data from {args.shard_dir}")
    volumes = get_validation_data(args.shard_dir, sae_data['config']['img_size'], args.batch_size, device)
    print(f"Loaded {volumes.shape[0]} volumes of shape {volumes.shape[1:]}")
    
    # Compute SAE activations
    print("Computing SAE activations...")
    tokens, activations = compute_sae_activations(mae, volumes, sae_weight, decoder_bias, sae_data['layer'], device)
    
    print(f"Token shape: {tokens.shape}")
    print(f"Activation shape: {activations.shape}")
    
    # Analyze statistics
    print("Analyzing latent statistics...")
    stats = analyze_latent_statistics(activations)
    
    print(f"Median sparsity: {np.median(stats['sparsity'])*100:.1f}%")
    print(f"Median active fraction: {np.median(stats['frac_active'])*100:.1f}%")
    
    # Create summary plots
    print("Creating summary plots...")
    create_summary_plots(stats, args.output_dir)
    
    # Find most active latents
    top_latent_indices = np.argsort(stats['total_mass'])[::-1][:args.num_latents]
    
    print(f"Visualizing top {args.num_latents} latents...")
    
    # Calculate patch grid shape
    img_size = sae_data['config']['img_size']
    patch_size = sae_data['config']['patch_size']
    grid_size = img_size // patch_size
    patch_grid_shape = (grid_size, grid_size, grid_size)
    
    # Visualize top latents
    for i, latent_idx in enumerate(top_latent_indices):
        print(f"Processing latent {latent_idx} ({i+1}/{args.num_latents})")
        
        # Get activations for this latent across all tokens
        latent_activations = activations[:, latent_idx]
        
        # Convert to 3D heatmap (using first volume's worth of tokens)
        tokens_per_vol = grid_size ** 3
        vol_activations = latent_activations[:tokens_per_vol]  # First volume
        
        heatmap = patch_coords_to_voxel_heat(vol_activations, patch_grid_shape, patch_size)
        
        # Get corresponding raw volume
        raw_volume = volumes[0, 0]  # First volume, single channel
        
        # Create visualization
        save_path = visualize_latent_on_volume(raw_volume, heatmap, latent_idx, args.output_dir)
        
        print(f"  Active fraction: {stats['frac_active'][latent_idx]*100:.2f}%")
        print(f"  Total mass: {stats['total_mass'][latent_idx]:.3f}")
        print(f"  Saved to: {save_path}")
    
    print(f"\nInspection complete! Results saved to {args.output_dir}")
    print(f"Key files:")
    print(f"  - sae_summary_stats.png: Overall SAE statistics")
    print(f"  - latent_XXXX_overlay.png: Individual latent visualizations")

if __name__ == "__main__":
    main() 