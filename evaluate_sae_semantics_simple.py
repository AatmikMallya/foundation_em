#!/usr/bin/env python3
"""
Simplified SAE Semantic Analysis - FAST VERSION with Optional Patch Visualization
================================================================================
Optimized for speed with vectorized operations and simplified correlation computation.
Optionally generates patch visualizations for top semantically selective neurons.
"""

import argparse
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.stats import pearsonr
from tqdm import tqdm

# Project imports
from vit_3d import mae_vit_3d_base_conv, get_device
from sae_train import LinearSAE, extract_patch_tokens

def flush_print(msg):
    """Print with immediate flush for SLURM logging"""
    print(msg, flush=True)
    sys.stdout.flush()

def load_sae_checkpoint(ckpt_path):
    """Load SAE checkpoint and extract key parameters"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    return {
        'weight': ckpt.get('sae_weight'),
        'encoder_weight': ckpt.get('encoder_weight'),
        'decoder_weight': ckpt.get('decoder_weight'),
        'encoder_bias': ckpt.get('encoder_bias'),
        'decoder_bias': ckpt.get('decoder_bias', None),
        'input_dim': ckpt['input_dim'], 
        'latent_dim': ckpt['latent_dim'],
        'layer': ckpt['layer'],
        'l1_coeff': ckpt['l1_coeff'],
        'config': ckpt['config'],
        'model_state_dict': ckpt.get('model_state_dict')
    }

def load_mae_model(config, device):
    """Load the pretrained MAE model"""
    mae = mae_vit_3d_base_conv(
        volume_size=(config['img_size'],) * 3,
        patch_size=config['patch_size'],
        mask_ratio=0.0
    ).to(device)
    
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

def load_samples_from_tar(tar_path, num_samples=10):
    """Load a small number of samples from a tar file"""
    samples = []
    with tarfile.open(tar_path, 'r') as tar:
        count = 0
        for member in tar:
            if not member.isfile():
                continue
            if count >= num_samples:
                break
                
            buf = tar.extractfile(member).read()
            
            if member.name.endswith('.bin') and len(buf) == 96**3:  # mask file
                data = np.frombuffer(buf, dtype=np.uint8).reshape(96, 96, 96).copy()
            else:  # volume file
                data = np.frombuffer(buf, dtype=np.float32).reshape(96, 96, 96).copy()
            
            samples.append(torch.from_numpy(data))
            count += 1
    
    return samples

def compute_patch_labels_vectorized(mask_batch):
    """Fast vectorized patch label computation using presence-based labeling"""
    B = mask_batch.shape[0]
    patch_size = 8
    patches_per_dim = 96 // patch_size  # 12
    
    # Reshape to patches: (B, 12, 12, 12, 8, 8, 8) 
    mask_patches = mask_batch.view(B, patches_per_dim, patch_size, 
                                 patches_per_dim, patch_size,
                                 patches_per_dim, patch_size)
    mask_patches = mask_patches.permute(0, 1, 3, 5, 2, 4, 6)
    mask_patches = mask_patches.reshape(B, patches_per_dim**3, patch_size**3)
    
    # Presence-based labeling: check if any voxels of each type are present
    # mask_patches shape: (B, 1728, 512)
    
    # Count occurrences of each label in each patch
    patch_has_membrane = (mask_patches == 1).any(dim=-1)  # (B, 1728)
    patch_has_sphere = (mask_patches == 2).any(dim=-1)    # (B, 1728)
    patch_has_cube = (mask_patches == 3).any(dim=-1)      # (B, 1728)
    
    # Create multi-hot encoding (patches can have multiple labels)
    multi_labels = torch.stack([
        patch_has_membrane,
        patch_has_sphere, 
        patch_has_cube
    ], dim=-1)  # (B, 1728, 3)
    
    # For backward compatibility, also create single labels with priority:
    # Priority: cube > sphere > membrane > background
    single_labels = torch.zeros(B, patches_per_dim**3, dtype=torch.long)
    single_labels[patch_has_membrane] = 1  # membrane
    single_labels[patch_has_sphere] = 2    # sphere (overwrites membrane) 
    single_labels[patch_has_cube] = 3      # cube (overwrites sphere/membrane)
    
    return {
        'single': single_labels.view(-1),  # (B*1728,) - for backward compatibility
        'multi': multi_labels.view(-1, 3),  # (B*1728, 3) - new multi-hot format
        'membrane': patch_has_membrane.view(-1),  # (B*1728,) - pure membrane patches
        'sphere': patch_has_sphere.view(-1),      # (B*1728,) - pure sphere patches  
        'cube': patch_has_cube.view(-1)           # (B*1728,) - pure cube patches
    }

def compute_correlations_fast(activations, labels, n_classes=4):
    """Memory-optimized correlation computation using matrix operations"""
    n_neurons = activations.shape[1]
    correlations = {}
    
    # Move to float32 if needed to save memory
    if activations.dtype == torch.float64:
        activations = activations.float()
    
    # Pre-compute statistics to avoid recomputation
    act_mean = activations.mean(dim=0, keepdim=True)
    act_std = activations.std(dim=0, keepdim=True)
    
    for class_idx in range(n_classes):
        # Create binary indicator for this class
        class_indicator = (labels == class_idx).float()
        
        # Skip if not enough samples
        if class_indicator.sum() < 10:
            correlations[class_idx] = []
            continue
        
        # Compute correlation using torch operations
        class_mean = class_indicator.mean()
        class_std = class_indicator.std()
        
        # Center the data efficiently
        class_centered = class_indicator - class_mean
        
        # Process in chunks to avoid large memory allocation
        chunk_size = 50000  # Larger chunks for efficiency
        correlations_list = []
        
        for start_idx in range(0, n_neurons, chunk_size):
            end_idx = min(start_idx + chunk_size, n_neurons)
            
            # Get subset of neurons
            act_subset = activations[:, start_idx:end_idx]
            act_centered_subset = act_subset - act_mean[:, start_idx:end_idx]
            
            # Compute covariance for this chunk
            covariance = (act_centered_subset * class_centered.unsqueeze(1)).mean(dim=0)
            
            # Compute correlations (avoid division by zero)
            act_std_subset = act_std[:, start_idx:end_idx].squeeze(0)
            correlations_tensor = covariance / (act_std_subset * class_std + 1e-8)
            
            # Convert to list with global neuron indices
            for local_idx, corr in enumerate(correlations_tensor):
                global_neuron_idx = start_idx + local_idx
                corr_val = corr.item()
                if not torch.isnan(torch.tensor(corr_val)) and abs(corr_val) > 0.01:
                    # Simple p-value approximation for large N
                    n = len(labels)
                    t_stat = corr_val * np.sqrt((n - 2) / (1 - corr_val**2 + 1e-8))
                    p_val = 2 * (1 - abs(t_stat) / np.sqrt(2 * np.pi))
                    correlations_list.append((global_neuron_idx, corr_val, max(p_val, 1e-10)))
            
            # Clean up memory for this chunk
            del act_subset, act_centered_subset, covariance, correlations_tensor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        correlations[class_idx] = correlations_list
        
        # Clean up memory
        del class_centered
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return correlations

def compute_shape_correlations(activations, shape_presence_labels, shape_name):
    """Compute correlations for neurons that respond to presence of specific shapes"""
    n_neurons = activations.shape[1]
    
    # Move to float32 if needed to save memory
    if activations.dtype == torch.float64:
        activations = activations.float()
    
    # Get binary indicator for shape presence
    shape_indicator = shape_presence_labels.float()
    
    # Skip if not enough positive samples
    n_positive = shape_indicator.sum().item()
    if n_positive < 20:  # Need at least 20 patches with this shape
        return []
    
    # Compute correlation using torch operations
    shape_mean = shape_indicator.mean()
    shape_std = shape_indicator.std()
    
    if shape_std < 1e-8:  # No variation
        return []
    
    # Pre-compute activation statistics
    act_mean = activations.mean(dim=0, keepdim=True)
    act_std = activations.std(dim=0, keepdim=True)
    
    # Center the data
    shape_centered = shape_indicator - shape_mean
    
    # Process in chunks to avoid large memory allocation
    chunk_size = 50000
    correlations_list = []
    
    for start_idx in range(0, n_neurons, chunk_size):
        end_idx = min(start_idx + chunk_size, n_neurons)
        
        # Get subset of neurons
        act_subset = activations[:, start_idx:end_idx]
        act_centered_subset = act_subset - act_mean[:, start_idx:end_idx]
        
        # Compute covariance for this chunk
        covariance = (act_centered_subset * shape_centered.unsqueeze(1)).mean(dim=0)
        
        # Compute correlations (avoid division by zero)
        act_std_subset = act_std[:, start_idx:end_idx].squeeze(0)
        correlations_tensor = covariance / (act_std_subset * shape_std + 1e-8)
        
        # Convert to list with global neuron indices
        for local_idx, corr in enumerate(correlations_tensor):
            global_neuron_idx = start_idx + local_idx
            corr_val = corr.item()
            if not torch.isnan(torch.tensor(corr_val)) and abs(corr_val) > 0.01:
                # Simple p-value approximation for large N
                n = len(shape_presence_labels)
                t_stat = corr_val * np.sqrt((n - 2) / (1 - corr_val**2 + 1e-8))
                p_val = 2 * (1 - abs(t_stat) / np.sqrt(2 * np.pi))
                correlations_list.append((global_neuron_idx, corr_val, max(p_val, 1e-10)))
        
        # Clean up memory for this chunk
        del act_subset, act_centered_subset, covariance, correlations_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Sort by absolute correlation
    correlations_list.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Clean up memory
    del shape_centered
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return correlations_list

def visualize_top_activating_patches(volumes, masks, all_activations, all_labels, neuron_idx, class_name, correlation, p_value, output_dir, top_k=16):
    """Visualize patches that most strongly activate a given neuron"""
    # Get activations for this neuron
    neuron_activations = all_activations[:, neuron_idx]
    
    # Get top activating patches
    top_indices = torch.argsort(neuron_activations, descending=True)[:top_k]
    
    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f'Top {top_k} Patches Activating Neuron {neuron_idx} ({class_name})\n'
                f'Correlation: r={correlation:.3f}, p={p_value:.2e}', fontsize=14)
    
    patch_size = 8
    patches_per_dim = 12
    
    # Count labels in the top activating patches
    top_patch_labels = all_labels[top_indices]
    label_counts = torch.bincount(top_patch_labels, minlength=4)
    label_names = ['Background', 'Membrane', 'Sphere', 'Cube']
    
    for i, global_patch_idx in enumerate(top_indices):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        
        # Find which volume and patch within that volume
        batch_idx = global_patch_idx // (patches_per_dim**3)
        patch_idx_in_batch = global_patch_idx % (patches_per_dim**3)
        
        if batch_idx < len(volumes):
            # Extract 3D patch coordinates
            z = patch_idx_in_batch // (patches_per_dim * patches_per_dim)
            y = (patch_idx_in_batch % (patches_per_dim * patches_per_dim)) // patches_per_dim
            x = patch_idx_in_batch % patches_per_dim
            
            # Extract the actual patch from volume
            volume = volumes[batch_idx]
            mask = masks[batch_idx]
            
            # Get patch boundaries
            x_start, x_end = x * patch_size, (x + 1) * patch_size
            y_start, y_end = y * patch_size, (y + 1) * patch_size
            z_start, z_end = z * patch_size, (z + 1) * patch_size
            
            # Extract patch (take middle slice for visualization)
            z_mid = (z_start + z_end) // 2
            patch_slice = volume[z_mid, y_start:y_end, x_start:x_end]
            mask_slice = mask[z_mid, y_start:y_end, x_start:x_end]
            
            # Create overlay visualization
            if patch_slice.max() > 0:
                patch_rgb = plt.cm.gray(patch_slice / patch_slice.max())[:, :, :3]
            else:
                patch_rgb = np.zeros((patch_slice.shape[0], patch_slice.shape[1], 3))
            
            # Color-code the mask overlay
            mask_colors = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # black, red, green, blue
            mask_overlay = mask_colors[mask_slice.long()]
            
            # Blend volume and mask
            alpha = 0.3
            blended = (1 - alpha) * patch_rgb + alpha * mask_overlay
            
            ax.imshow(blended)
            
            activation_val = neuron_activations[global_patch_idx].item()
            patch_label = all_labels[global_patch_idx].item()
            
            ax.set_title(f'Act: {activation_val:.3f}\nLabel: {label_names[patch_label]}\nCoord: ({x},{y},{z})', fontsize=10)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Add text box with statistics
    stats_text = f"Top {top_k} patch labels:\n"
    for i, (name, count) in enumerate(zip(label_names, label_counts)):
        pct = 100 * count / top_k
        stats_text += f"{name}: {count} ({pct:.1f}%)\n"
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / f'neuron_{neuron_idx}_{class_name}_top_patches.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Simplified SAE semantic analysis with optional patch visualization")
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--volume_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sae_semantic_analysis_simple")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--extract_from", choices=["patchembed", "encoder"], default=None,
                       help="Override extraction method (auto-detected from checkpoint if not specified)")
    parser.add_argument("--layer", type=int, default=None,
                       help="Specify encoder layer (used if extract_from is 'encoder')")
    parser.add_argument("--visualize_patches", action="store_true", 
                       help="Generate patch visualizations for top semantically selective neurons")
    parser.add_argument("--top_neurons", type=int, default=3, 
                       help="Number of top neurons per class to visualize (only used with --visualize_patches)")
    
    args = parser.parse_args()
    
    flush_print("=== Fast SAE Semantic Analysis Starting ===")
    
    device = get_device()
    flush_print(f"Using device: {device}")
    
    # Load SAE checkpoint
    flush_print(f"Loading SAE checkpoint from {args.sae_checkpoint}")
    sae_data = load_sae_checkpoint(args.sae_checkpoint)
    flush_print(f"SAE info: {sae_data['latent_dim']} latents, layer {sae_data['layer']}")
    
    # Create organized output directory structure
    # Extract checkpoint name from path (without .pt extension)
    checkpoint_name = Path(args.sae_checkpoint).stem
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / "checkpoints" / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    flush_print(f"Results will be saved to: {output_dir}")
    
    # Determine extraction method: command line arg > checkpoint config > default
    if args.extract_from is not None:
        extract_from = args.extract_from
        flush_print(f"Using command-line specified extraction method: {extract_from}")
    else:
        sae_config = sae_data.get('config', {})
        extract_from = sae_config.get('extract_from', 'encoder')  # Default to encoder for old checkpoints
        flush_print(f"Auto-detected extraction method from checkpoint: {extract_from}")
    
    # Determine layer to use
    if extract_from == 'encoder':
        layer_idx = args.layer if args.layer is not None else sae_data['layer']
        flush_print(f"Extracting from encoder layer {layer_idx}")
    else:
        layer_idx = sae_data['layer'] # Pass along for consistency, even if unused by extractor
        flush_print("Extracting from PatchEmbed output (raw CNN features)")
    
    # Load MAE model
    flush_print("Loading MAE model...")
    mae = load_mae_model(sae_data['config'], device)
    
    # Load SAE
    flush_print("Loading SAE...")
    if sae_data.get('model_state_dict') is not None:
        sae = LinearSAE(
            input_dim=sae_data['input_dim'],
            latent_dim=sae_data['latent_dim']
        )
        sae.load_state_dict(sae_data['model_state_dict'])
    else:
        # Old format - create minimal wrapper
        class OldFormatSAE(torch.nn.Module):
            def __init__(self, weight, decoder_bias=None):
                super().__init__()
                if weight is not None:
                    self.register_buffer('weight', weight)
                    self.register_buffer('encoder_weight', weight)
                else:
                    raise ValueError("No valid SAE weights found in checkpoint")
                if decoder_bias is not None:
                    self.register_buffer('decoder_bias', decoder_bias)
                else:
                    self.decoder_bias = None
            
            def eval(self):
                return self
        
        sae = OldFormatSAE(sae_data['weight'], sae_data['decoder_bias'])
    
    sae = sae.to(device)
    sae.eval()
    
    # Load samples
    num_samples_for_patches = args.num_samples if args.visualize_patches else args.num_samples
    flush_print(f"Loading {num_samples_for_patches} volume samples...")
    volume_shards = sorted(Path(args.volume_dir).glob("shard_*.tar"))
    volumes = load_samples_from_tar(volume_shards[0], num_samples_for_patches)
    
    flush_print(f"Loading {num_samples_for_patches} mask samples...")
    mask_shards = sorted(Path(args.mask_dir).glob("shard_*.tar"))
    masks = load_samples_from_tar(mask_shards[0], num_samples_for_patches)
    
    flush_print(f"Loaded {len(volumes)} volumes and {len(masks)} masks")
    
    # Process samples efficiently
    batch_size = 32  # Larger batch size for efficiency
    
    # Collect all activations and labels (including shape-specific data)
    all_activations = []
    all_labels = []
    all_shape_labels = {'membrane': [], 'sphere': [], 'cube': []}
    
    flush_print("Processing volumes...")
    with torch.no_grad():
        for i in tqdm(range(0, min(len(volumes), len(masks)), batch_size), desc="Processing batches"):
            # Get batch
            vol_batch = torch.stack(volumes[i:i+batch_size]).unsqueeze(1).to(device)
            mask_batch = torch.stack(masks[i:i+batch_size]).to(device)
            
            # Extract tokens using the method that matches the SAE training
            tokens = extract_patch_tokens(mae, vol_batch, layer_idx, extract_from=extract_from)
            
            # Forward through SAE
            if hasattr(sae, 'encode') and callable(getattr(sae, 'encode')):
                sae_recon, sae_activations = sae(tokens)
            else:
                # Old format
                if hasattr(sae, 'decoder_bias') and sae.decoder_bias is not None:
                    tokens_centered = tokens - sae.decoder_bias
                    sae_weight = sae.encoder_weight
                    sae_activations = torch.relu(tokens_centered @ sae_weight.T)
                else:
                    sae_weight = sae.weight
                    sae_activations = torch.relu(tokens @ sae_weight.T)
            
            # Fast patch label computation (now returns dict with multiple formats)
            patch_labels = compute_patch_labels_vectorized(mask_batch)
            
            # Store results (move to CPU to save GPU memory)
            all_activations.append(sae_activations.cpu())
            all_labels.append(patch_labels['single'].cpu())
            
            # Store shape-specific presence labels
            all_shape_labels['membrane'].append(patch_labels['membrane'].cpu())
            all_shape_labels['sphere'].append(patch_labels['sphere'].cpu())
            all_shape_labels['cube'].append(patch_labels['cube'].cpu())
            
            # Clear GPU memory
            del vol_batch, mask_batch, tokens, sae_activations
            torch.cuda.empty_cache()
    
    # Concatenate all results
    flush_print("Computing correlations...")
    all_activations = torch.cat(all_activations, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Concatenate shape-specific labels
    for shape_name in all_shape_labels:
        all_shape_labels[shape_name] = torch.cat(all_shape_labels[shape_name], dim=0)
    
    flush_print(f"Data collected: {all_activations.shape[0]} patches, {all_activations.shape[1]} neurons")
    
    total_patches = len(all_labels)
    label_counts = torch.bincount(all_labels, minlength=4)
    
    flush_print(f"Final results: {total_patches} patches processed")
    flush_print(f"Label distribution: {label_counts}")
    
    # Report label percentages
    label_percentages = 100 * label_counts.float() / total_patches
    class_names = ['Background', 'Membrane', 'Sphere', 'Cube']
    for i, (name, count, pct) in enumerate(zip(class_names, label_counts, label_percentages)):
        flush_print(f"  {name}: {count} patches ({pct:.1f}%)")
    
    # Fast correlation computation
    semantic_classes = {
        'background': 0,
        'membrane': 1, 
        'sphere': 2,
        'cube': 3
    }
    
    all_class_correlations = {}
    for class_name, class_label in semantic_classes.items():
        flush_print(f"Computing correlations for {class_name}...")
        correlations = compute_correlations_fast(all_activations, all_labels == class_label)
        correlations[class_label].sort(key=lambda x: abs(x[1]), reverse=True)
        all_class_correlations[class_name] = correlations[class_label]
    
    # NEW: Shape-specific presence analysis
    flush_print("\n=== Shape Presence Analysis ===")
    shape_presence_correlations = {}
    
    for shape_name in ['membrane', 'sphere', 'cube']:
        flush_print(f"Computing shape presence correlations for {shape_name}...")
        shape_labels = all_shape_labels[shape_name]
        n_present = shape_labels.sum().item()
        n_total = len(shape_labels)
        
        flush_print(f"  {shape_name}: {n_present}/{n_total} patches ({100*n_present/n_total:.1f}%) contain this shape")
        
        if n_present >= 20:  # Need sufficient positive examples
            correlations = compute_shape_correlations(all_activations, shape_labels, shape_name)
            shape_presence_correlations[shape_name] = correlations
            
            # Print top shape-selective neurons
            if correlations:
                flush_print(f"  Top 10 {shape_name}-presence neurons:")
                for i, (neuron_idx, corr, p_val) in enumerate(correlations[:10]):
                    flush_print(f"    {i+1:2d}. Neuron {neuron_idx:4d}: r={corr:6.3f}, p={p_val:.3e}")
        else:
            flush_print(f"  Skipping {shape_name}: insufficient patches (need ≥20, have {n_present})")
            shape_presence_correlations[shape_name] = []
    
    # Print analysis for each class
    for class_name, class_label in semantic_classes.items():
        n_patches = label_counts[class_label].item()
        flush_print(f"\nAnalyzing {class_name} class (label {class_label}): {n_patches} patches")
        
        if all_class_correlations[class_name]:
            # Print top 20 neurons for this class
            flush_print(f"Top 20 {class_name}-selective neurons:")
            for i, (neuron_idx, corr, p_val) in enumerate(all_class_correlations[class_name][:20]):
                flush_print(f"  {i+1:2d}. Neuron {neuron_idx:4d}: r={corr:6.3f}, p={p_val:.3e}")
        else:
            flush_print(f"No correlations found for {class_name} class (had {n_patches} patches)")
            if n_patches < 100:
                flush_print(f"  Reason: Too few patches (need ≥100, had {n_patches})")
            else:
                flush_print(f"  Reason: All correlations were invalid/NaN or filtered out")
    
    # Create comprehensive visualizations
    has_any_correlations = (
        any(all_class_correlations.values()) or 
        any(shape_presence_correlations.values())
    )
    
    if has_any_correlations:
        # Plot 1: Correlation distributions for each class (use best available analysis)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SAE Neuron Correlations by Semantic Class', fontsize=16)
        
        colors = ['blue', 'green', 'red', 'orange']
        class_names = list(semantic_classes.keys())
        
        for i, (class_name, ax) in enumerate(zip(class_names, axes.flat)):
            # Use traditional analysis if available, otherwise use presence-based
            if all_class_correlations[class_name]:
                corrs = [abs(c[1]) for c in all_class_correlations[class_name]]
                analysis_type = "Traditional"
            elif class_name in shape_presence_correlations and shape_presence_correlations[class_name]:
                corrs = [abs(c[1]) for c in shape_presence_correlations[class_name]]
                analysis_type = "Presence-Based"
            else:
                corrs = []
                analysis_type = "No Data"
            
            if corrs:
                ax.hist(corrs, bins=30, alpha=0.7, color=colors[i], edgecolor='black')
                ax.set_title(f'{class_name.title()} Correlations ({analysis_type})')
                ax.set_xlabel('|Correlation|')
                ax.set_ylabel('Number of neurons')
                ax.axvline(0.1, color='red', linestyle='--', alpha=0.7)
                ax.axvline(0.2, color='orange', linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, f'Insufficient\n{class_name} patches', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{class_name.title()} Correlations')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'all_class_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Top neurons comparison across classes (use best available analysis)
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(class_names):
            # Use traditional analysis if available, otherwise use presence-based
            if all_class_correlations[class_name]:
                top_neurons = all_class_correlations[class_name][:15]
                label_suffix = ""
            elif class_name in shape_presence_correlations and shape_presence_correlations[class_name]:
                top_neurons = shape_presence_correlations[class_name][:15]
                label_suffix = " (presence)"
            else:
                continue
                
            corr_vals = [n[1] for n in top_neurons]  # Keep sign for comparison
            y_pos = np.arange(len(corr_vals)) + i * 0.2
            plt.barh(y_pos, corr_vals, height=0.15, label=f"{class_name}{label_suffix}", 
                    color=colors[i], alpha=0.7)
        
        plt.title('Top 15 Neurons per Semantic Class (with sign)')
        plt.xlabel('Correlation')
        plt.ylabel('Neuron rank')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'top_neurons_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Shape presence vs majority vote comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Shape Detection: Presence-Based vs Majority Vote Analysis', fontsize=16)
        
        shape_names = ['membrane', 'sphere', 'cube']
        traditional_names = ['membrane', 'sphere', 'cube']  # corresponding traditional analysis
        
        for i, (shape_name, trad_name) in enumerate(zip(shape_names, traditional_names)):
            # Top plot: Presence-based analysis
            ax_top = axes[0, i]
            if shape_name in shape_presence_correlations and shape_presence_correlations[shape_name]:
                top_neurons = shape_presence_correlations[shape_name][:15]
                corr_vals = [abs(n[1]) for n in top_neurons]  # Use absolute values for comparison
                neuron_indices = [n[0] for n in top_neurons]
                
                bars = ax_top.bar(range(len(corr_vals)), corr_vals, color='darkgreen', alpha=0.7)
                ax_top.set_title(f'{shape_name.title()} Presence Detection\n({len(shape_presence_correlations[shape_name])} responsive neurons)')
                ax_top.set_ylabel('|Correlation|')
                ax_top.set_ylim(0, max(0.3, max(corr_vals) * 1.1) if corr_vals else 0.3)
                
                # Add neuron indices as labels
                for j, (bar, neuron_idx) in enumerate(zip(bars, neuron_indices)):
                    if j < 5:  # Only label top 5 for readability
                        ax_top.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                  f'{neuron_idx}', ha='center', va='bottom', fontsize=8, rotation=45)
            else:
                ax_top.text(0.5, 0.5, f'No {shape_name}\ndata', ha='center', va='center', transform=ax_top.transAxes)
                ax_top.set_title(f'{shape_name.title()} Presence Detection')
            
            # Bottom plot: Traditional majority-vote analysis 
            ax_bottom = axes[1, i]
            if trad_name in all_class_correlations and all_class_correlations[trad_name]:
                top_neurons = all_class_correlations[trad_name][:15]
                corr_vals = [abs(n[1]) for n in top_neurons]
                neuron_indices = [n[0] for n in top_neurons]
                
                bars = ax_bottom.bar(range(len(corr_vals)), corr_vals, color='darkblue', alpha=0.7)
                ax_bottom.set_title(f'{trad_name.title()} Majority Vote\n({len(all_class_correlations[trad_name])} responsive neurons)')
                ax_bottom.set_ylabel('|Correlation|')
                ax_bottom.set_xlabel('Neuron Rank')
                ax_bottom.set_ylim(0, max(0.3, max(corr_vals) * 1.1) if corr_vals else 0.3)
                
                # Add neuron indices as labels
                for j, (bar, neuron_idx) in enumerate(zip(bars, neuron_indices)):
                    if j < 5:  # Only label top 5 for readability
                        ax_bottom.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                     f'{neuron_idx}', ha='center', va='bottom', fontsize=8, rotation=45)
            else:
                ax_bottom.text(0.5, 0.5, f'No {trad_name}\ndata', ha='center', va='center', transform=ax_bottom.transAxes)
                ax_bottom.set_title(f'{trad_name.title()} Majority Vote')
                ax_bottom.set_xlabel('Neuron Rank')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'presence_vs_majority_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save comprehensive results
        with open(output_dir / 'all_class_results.txt', 'w') as f:
            f.write(f"Complete SAE Semantic Analysis Results\n")
            f.write(f"=====================================\n\n")
            f.write(f"SAE Checkpoint: {args.sae_checkpoint}\n")
            f.write(f"Checkpoint Name: {checkpoint_name}\n")
            f.write(f"Processed {total_patches} patches\n")
            f.write(f"SAE latent dimension: {sae_data['latent_dim']}\n")
            f.write(f"Extraction method: {extract_from}\n")
            f.write(f"Layer: {sae_data['layer']}\n\n")
            
            # Traditional majority-vote analysis
            f.write("TRADITIONAL ANALYSIS (Majority Vote)\n")
            f.write("=" * 40 + "\n\n")
            
            for class_name, class_label in semantic_classes.items():
                n_patches = label_counts[class_label].item()
                f.write(f"{class_name.upper()} (label {class_label}): {n_patches} patches\n")
                f.write("-" * 50 + "\n")
                
                if all_class_correlations[class_name]:
                    f.write(f"Top 20 {class_name}-selective neurons:\n")
                    for i, (neuron_idx, corr, p_val) in enumerate(all_class_correlations[class_name][:20]):
                        f.write(f"  {i+1:2d}. Neuron {neuron_idx:4d}: r={corr:6.3f}, p={p_val:.3e}\n")
                else:
                    f.write(f"Insufficient patches for correlation analysis\n")
                f.write("\n")
            
            # NEW: Shape presence analysis
            f.write("\nSHAPE PRESENCE ANALYSIS (Any Shape Present)\n")
            f.write("=" * 45 + "\n\n")
            
            for shape_name in ['membrane', 'sphere', 'cube']:
                shape_labels = all_shape_labels[shape_name]
                n_present = shape_labels.sum().item()
                n_total = len(shape_labels)
                
                f.write(f"{shape_name.upper()} PRESENCE: {n_present}/{n_total} patches ({100*n_present/n_total:.1f}%)\n")
                f.write("-" * 50 + "\n")
                
                if shape_name in shape_presence_correlations and shape_presence_correlations[shape_name]:
                    f.write(f"Top 20 {shape_name}-presence neurons:\n")
                    for i, (neuron_idx, corr, p_val) in enumerate(shape_presence_correlations[shape_name][:20]):
                        f.write(f"  {i+1:2d}. Neuron {neuron_idx:4d}: r={corr:6.3f}, p={p_val:.3e}\n")
                else:
                    f.write(f"Insufficient patches for correlation analysis (need ≥20, have {n_present})\n")
                f.write("\n")

        # Generate patch visualizations if requested
        if args.visualize_patches:
            flush_print("\n=== Generating patch visualizations ===")
            patch_dir = output_dir / "patch_visualizations"
            patch_dir.mkdir(parents=True, exist_ok=True)
            
            total_visualizations = 0
            
            # For each class, use the best available analysis method
            classes_to_process = [
                ('background', 'background'),
                ('membrane', 'membrane'), 
                ('sphere', 'sphere'),
                ('cube', 'cube')
            ]
            
            for display_name, analysis_key in classes_to_process:
                # Choose the best available correlation source
                if analysis_key in all_class_correlations and len(all_class_correlations[analysis_key]) > 0:
                    correlations_source = all_class_correlations[analysis_key]
                    source_type = "traditional"
                elif analysis_key in shape_presence_correlations and len(shape_presence_correlations[analysis_key]) > 0:
                    correlations_source = shape_presence_correlations[analysis_key]
                    source_type = "presence-based"
                else:
                    flush_print(f"Skipping {display_name}: no correlation data available")
                    continue
                
                flush_print(f"Processing {display_name} class (using {source_type} analysis)...")
                
                # Get top correlated neurons for this class
                top_neurons = correlations_source[:args.top_neurons]
                
                # Generate visualizations for top neurons (clean file names)
                for i, (neuron_idx, correlation, p_value) in enumerate(top_neurons):
                    flush_print(f"  Generating patches for {display_name} neuron {neuron_idx} (rank {i+1}, r={correlation:.3f})")
                    visualize_top_activating_patches(
                        volumes, masks, all_activations, all_labels, 
                        neuron_idx, display_name, correlation, p_value, patch_dir
                    )
                    total_visualizations += 1
            
            # Create a summary file with neuron information
            summary_file = patch_dir / "neuron_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("Patch Visualization Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Total visualizations generated: {total_visualizations}\n")
                f.write(f"Neurons analyzed per class: {args.top_neurons}\n\n")
                
                for display_name, analysis_key in classes_to_process:
                    # Find which analysis method was used
                    if analysis_key in all_class_correlations and len(all_class_correlations[analysis_key]) > 0:
                        correlations_source = all_class_correlations[analysis_key]
                        source_desc = "Traditional majority-vote analysis"
                    elif analysis_key in shape_presence_correlations and len(shape_presence_correlations[analysis_key]) > 0:
                        correlations_source = shape_presence_correlations[analysis_key]
                        source_desc = "Presence-based analysis"
                    else:
                        continue
                    
                    f.write(f"{display_name.upper()} neurons ({source_desc}):\n")
                    f.write("-" * 50 + "\n")
                    
                    top_neurons = correlations_source[:args.top_neurons]
                    for i, (neuron_idx, correlation, p_value) in enumerate(top_neurons):
                        f.write(f"  {i+1}. Neuron {neuron_idx:4d}: r={correlation:6.3f}, p={p_value:.2e}\n")
                        f.write(f"     File: neuron_{neuron_idx}_{display_name}_top_patches.png\n")
                    f.write("\n")
            
            flush_print(f"Patch visualizations complete!")
            flush_print(f"Patch results saved to {patch_dir}/")
            flush_print(f"Generated {total_visualizations} visualization files")
            flush_print(f"Patch summary saved to {summary_file}")

        flush_print(f"\n=== Analysis Complete for {checkpoint_name} ===")
        flush_print(f"Results saved to: {output_dir}/")
        flush_print(f"Full path: {output_dir.absolute()}")
        flush_print("Generated files:")
        flush_print("  - all_class_correlations.png: Correlation histograms for each class")
        flush_print("  - top_neurons_comparison.png: Top neurons across all classes")
        flush_print("  - presence_vs_majority_comparison.png: Shape presence vs majority vote analysis")
        flush_print("  - all_class_results.txt: Detailed results for traditional AND presence-based analysis")
        if args.visualize_patches:
            flush_print("  - patch_visualizations/: Individual neuron activation patches")
            flush_print(f"    * Generated {total_visualizations} patch visualization files")
    else:
        flush_print("No classes had sufficient patches for analysis")
        flush_print(f"Results directory created: {output_dir}")
    
    flush_print(f"\n=== Directory Structure ===")
    flush_print(f"Base: {base_output_dir}")
    flush_print(f"Checkpoint-specific: checkpoints/{checkpoint_name}/")
    flush_print("=== Analysis complete! ===")

if __name__ == "__main__":
    main() 