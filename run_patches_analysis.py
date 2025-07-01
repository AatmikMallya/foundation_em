#!/usr/bin/env python3
"""
Generate patch visualizations for top semantically selective SAE neurons
Reads correlation results from semantic evaluation and creates patch activation images

KEY IMPROVEMENTS:
1. Uses SEMANTICALLY CORRELATED neurons (not just highest activating)
2. Reads correlation results from evaluate_sae_semantics_simple.py output
3. Includes correlation statistics (r-value, p-value) in visualizations
4. Outputs to subdirectory of semantic analysis results
5. Shows label distribution statistics for top activating patches
6. Proper flush_print for SLURM logging

WORKFLOW:
1. Run evaluate_sae_semantics_simple.py first to get correlations
2. This script loads those correlation results 
3. For each class, visualizes patches for the most correlated neurons
4. Results saved to [semantic_results_dir]/patch_visualizations/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import tarfile
from tqdm import tqdm
import re

# Project imports
from vit_3d import mae_vit_3d_base_conv, get_device
from sae_train import LinearSAE, extract_patch_tokens

def flush_print(msg):
    """Print with immediate flush for SLURM logging"""
    print(msg, flush=True)

def load_correlation_results(results_file):
    """Load correlation results from semantic evaluation output"""
    if not results_file.exists():
        raise FileNotFoundError(f"Correlation results not found: {results_file}")
    
    correlations = {}
    current_class = None
    
    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Look for class header like "BACKGROUND (label 0): 3456 patches"
            if ") patches" in line and "(label " in line:
                class_match = re.search(r'(\w+) \(label \d+\):', line)
                if class_match:
                    current_class = class_match.group(1).lower()
                    correlations[current_class] = []
            # Look for neuron lines like "  1. Neuron 2012: r= 0.387, p=1.00e-10"
            elif current_class and re.match(r'\s*\d+\.\s+Neuron\s+\d+:', line):
                match = re.search(r'Neuron\s+(\d+):\s+r=\s*([-+]?\d*\.?\d+),\s+p=\s*([\d\.e\-\+]+)', line)
                if match:
                    neuron_idx = int(match.group(1))
                    correlation = float(match.group(2))
                    p_value = float(match.group(3))
                    correlations[current_class].append((neuron_idx, correlation, p_value))
    
    return correlations

def load_samples_from_tar(tar_path, num_samples=300):
    """Load samples from a tar file"""
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

def load_sae_checkpoint(ckpt_path):
    """Load SAE checkpoint"""
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

def compute_patch_labels_vectorized(mask_batch):
    """Fast vectorized patch label computation"""
    B = mask_batch.shape[0]
    patch_size = 8
    patches_per_dim = 96 // patch_size  # 12
    
    # Reshape to patches
    mask_patches = mask_batch.view(B, patches_per_dim, patch_size, 
                                 patches_per_dim, patch_size,
                                 patches_per_dim, patch_size)
    mask_patches = mask_patches.permute(0, 1, 3, 5, 2, 4, 6)
    mask_patches = mask_patches.reshape(B, patches_per_dim**3, patch_size**3)
    
    # Take mode per patch
    patch_labels = torch.mode(mask_patches, dim=-1)[0]
    return patch_labels.view(-1)

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
    parser = argparse.ArgumentParser(description="Generate patch visualizations for top semantically selective SAE neurons")
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--volume_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--semantic_results_dir", type=str, default="sae_semantic_analysis_simple",
                       help="Directory containing semantic evaluation results")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--top_neurons", type=int, default=3, help="Number of top neurons per class to visualize")
    
    args = parser.parse_args()
    
    device = get_device()
    flush_print(f"Using device: {device}")
    
    # Set up output directory as subdirectory of semantic results
    semantic_dir = Path(args.semantic_results_dir)
    output_dir = semantic_dir / "patch_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    flush_print(f"Output directory: {output_dir}")
    
    # Load correlation results from semantic evaluation
    results_file = semantic_dir / "all_class_results.txt"
    flush_print(f"Loading correlation results from {results_file}")
    try:
        correlations = load_correlation_results(results_file)
        flush_print(f"Loaded correlations for classes: {list(correlations.keys())}")
    except FileNotFoundError as e:
        flush_print(f"ERROR: {e}")
        flush_print("Please run the semantic evaluation first using evaluate_sae_semantics_simple.py")
        return
    
    # Load SAE checkpoint
    flush_print(f"Loading SAE checkpoint from {args.sae_checkpoint}")
    sae_data = load_sae_checkpoint(args.sae_checkpoint)
    
    # Determine extraction method
    sae_config = sae_data.get('config', {})
    extract_from = sae_config.get('extract_from', 'encoder')
    flush_print(f"Using extraction method: {extract_from}")
    
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
    flush_print(f"Loading {args.num_samples} volume samples...")
    volume_shards = sorted(Path(args.volume_dir).glob("shard_*.tar"))
    volumes = load_samples_from_tar(volume_shards[0], args.num_samples)
    
    flush_print(f"Loading {args.num_samples} mask samples...")
    mask_shards = sorted(Path(args.mask_dir).glob("shard_*.tar"))
    masks = load_samples_from_tar(mask_shards[0], args.num_samples)
    
    flush_print(f"Loaded {len(volumes)} volumes and {len(masks)} masks")
    
    # Process samples to get activations and labels
    batch_size = 16
    layer_idx = sae_data['layer']
    
    all_activations = []
    all_labels = []
    
    flush_print("Processing volumes...")
    with torch.no_grad():
        for i in tqdm(range(0, min(len(volumes), len(masks)), batch_size), desc="Processing batches"):
            vol_batch = torch.stack(volumes[i:i+batch_size]).unsqueeze(1).to(device)
            mask_batch = torch.stack(masks[i:i+batch_size]).to(device)
            
            # Extract tokens
            tokens = extract_patch_tokens(mae, vol_batch, layer_idx, extract_from=extract_from)
            
            # Forward through SAE
            if hasattr(sae, 'encode') and callable(getattr(sae, 'encode')):
                sae_recon, sae_activations = sae(tokens)
            else:
                if hasattr(sae, 'decoder_bias') and sae.decoder_bias is not None:
                    tokens_centered = tokens - sae.decoder_bias
                    sae_weight = sae.encoder_weight
                    sae_activations = torch.relu(tokens_centered @ sae_weight.T)
                else:
                    sae_weight = sae.weight
                    sae_activations = torch.relu(tokens @ sae_weight.T)
            
            # Compute patch labels
            patch_labels = compute_patch_labels_vectorized(mask_batch)
            
            all_activations.append(sae_activations.cpu())
            all_labels.append(patch_labels.cpu())
            
            del vol_batch, mask_batch, tokens, sae_activations
            torch.cuda.empty_cache()
    
    # Concatenate results
    all_activations = torch.cat(all_activations, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    flush_print(f"Data collected: {all_activations.shape[0]} patches, {all_activations.shape[1]} neurons")
    
    # Generate visualizations for top correlated neurons
    flush_print("Generating patch visualizations for top semantically selective neurons...")
    
    total_visualizations = 0
    for class_name in ['background', 'membrane', 'sphere', 'cube']:
        if class_name not in correlations or len(correlations[class_name]) == 0:
            flush_print(f"Skipping {class_name}: no correlation data")
            continue
            
        flush_print(f"Processing {class_name} class...")
        
        # Get top correlated neurons for this class (sorted by absolute correlation)
        top_neurons = correlations[class_name][:args.top_neurons]
        
        # Generate visualizations for top neurons
        for i, (neuron_idx, correlation, p_value) in enumerate(top_neurons):
            flush_print(f"  Generating patches for {class_name} neuron {neuron_idx} (rank {i+1}, r={correlation:.3f})")
            visualize_top_activating_patches(
                volumes, masks, all_activations, all_labels, 
                neuron_idx, class_name, correlation, p_value, output_dir
            )
            total_visualizations += 1
    
    # Create a summary file with neuron information
    summary_file = output_dir / "neuron_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Patch Visualization Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total visualizations generated: {total_visualizations}\n")
        f.write(f"Neurons analyzed per class: {args.top_neurons}\n\n")
        
        for class_name in ['background', 'membrane', 'sphere', 'cube']:
            if class_name in correlations and correlations[class_name]:
                f.write(f"{class_name.upper()} neurons:\n")
                f.write("-" * 20 + "\n")
                
                top_neurons = correlations[class_name][:args.top_neurons]
                for i, (neuron_idx, correlation, p_value) in enumerate(top_neurons):
                    f.write(f"  {i+1}. Neuron {neuron_idx:4d}: r={correlation:6.3f}, p={p_value:.2e}\n")
                    f.write(f"     File: neuron_{neuron_idx}_{class_name}_top_patches.png\n")
                f.write("\n")
    
    flush_print(f"Patch visualizations complete!")
    flush_print(f"Results saved to {output_dir}/")
    flush_print(f"Generated {total_visualizations} visualization files")
    flush_print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main() 