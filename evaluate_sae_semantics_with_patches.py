#!/usr/bin/env python3
"""
Evaluate SAE semantic interpretability using ground truth segmentation masks
============================================================================

This script:
1. Loads a trained SAE and ViT model
2. Extracts SAE activations on validation volumes 
3. Loads corresponding segmentation masks
4. Computes correlations between SAE neurons and semantic features
5. Generates visualizations of semantic neuron responses
"""

import argparse
import io
import random
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm

# Project imports - use the same as inspect_sae.py
from vit_3d import mae_vit_3d_base_conv, get_device
from sae_train import LinearSAE, extract_patch_tokens
from vol_train import TarShardDataset

warnings.filterwarnings("ignore", category=UserWarning)

# AMP dtype mapping (bf16 recommended for H100)
_AMP_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def flush_print(msg):
    """Print with immediate flush for SLURM logging"""
    print(msg, flush=True)
    sys.stdout.flush()


def load_sae_checkpoint(ckpt_path):
    """Load SAE checkpoint and extract key parameters - same as inspect_sae.py"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    return {
        'weight': ckpt.get('sae_weight'),  # Old format
        'encoder_weight': ckpt.get('encoder_weight'),  # New format
        'decoder_weight': ckpt.get('decoder_weight'),  # New format
        'encoder_bias': ckpt.get('encoder_bias'),
        'decoder_bias': ckpt.get('decoder_bias', None),
        'input_dim': ckpt['input_dim'], 
        'latent_dim': ckpt['latent_dim'],
        'layer': ckpt['layer'],
        'l1_coeff': ckpt['l1_coeff'],
        'config': ckpt['config'],
        'model_state_dict': ckpt.get('model_state_dict')  # New format
    }


def load_mae_model(config, device):
    """Load the pretrained MAE model - same as inspect_sae.py"""
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


class MaskTarDataset(torch.utils.data.IterableDataset):
    """Dataset for loading segmentation masks from tar files - follows TarShardDataset pattern"""
    
    def __init__(self, mask_dir: Path, shard_pattern="shard_*.tar", shuffle=False):
        self.mask_files = sorted(mask_dir.glob(shard_pattern))
        if not self.mask_files:
            raise ValueError(f"No mask files found in {mask_dir}")
        
        self.shuffle = shuffle
        flush_print(f"Found {len(self.mask_files)} mask shards in {mask_dir}")
    
    def _iter_shard(self, path):
        """Iterate through masks in a single shard - similar to TarShardDataset"""
        try:
            with tarfile.open(path, "r|", bufsize=32*1024*1024) as tar:  # 32MB buffer like TarShardDataset
                for member in tar:
                    if not member.isfile() or not member.name.endswith('.bin'):
                        continue
                    
                    # Extract mask data
                    buf = tar.extractfile(member).read()
                    mask = np.frombuffer(buf, dtype=np.uint8).reshape(96, 96, 96)
                    
                    # Create tensor with proper memory layout (no CUDA operations in workers)
                    mask_tensor = torch.from_numpy(mask).contiguous()
                    yield mask_tensor
                    
        except Exception as e:
            flush_print(f"Error reading mask shard {path}: {e}")
            return
    
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        shard_list = self.mask_files.copy()
        if self.shuffle:
            random.shuffle(shard_list)
        
        # Distribute shards across workers with better load balancing
        if worker:
            # Round-robin distribution for better load balancing
            worker_shards = shard_list[worker.id::worker.num_workers]
        else:
            worker_shards = shard_list
            
        for shard_path in worker_shards:
            yield from self._iter_shard(shard_path)


def extract_sae_activations(model, sae, volume_dataloader, device, layer_idx, max_samples=None, use_amp=True, amp_dtype="bf16", mask_dataloader=None):
    """Extract SAE activations for all patches in the dataset - optimized version with H100 optimizations"""
    flush_print("Starting SAE activation extraction...")
    flush_print(f"Device: {device}, AMP: {use_amp}, dtype: {amp_dtype}")
    
    model.eval()
    sae.eval()
    
    # Set up mixed precision
    autocast_dtype = _AMP_DTYPE_MAP.get(amp_dtype, torch.bfloat16)
    flush_print(f"Using autocast dtype: {autocast_dtype}")
    
    all_activations = []
    all_mask_patches = []
    total_tokens = 0
    samples_processed = 0
    
    # Create iterators for both dataloaders
    flush_print("Creating volume iterator...")
    volume_iter = iter(volume_dataloader)
    flush_print("Creating mask iterator...")
    mask_iter = iter(mask_dataloader)
    
    with torch.no_grad():
        batch_idx = 0
        while True:
            if max_samples and samples_processed >= max_samples:
                flush_print(f"Reached max_samples limit ({max_samples}), stopping extraction")
                break
                
            try:
                flush_print(f"Getting volume batch {batch_idx + 1}...")
                volumes = next(volume_iter)
                flush_print(f"Got volume batch with shape: {volumes.shape}")
                
                flush_print(f"Getting mask batch {batch_idx + 1}...")
                masks = next(mask_iter)
                flush_print(f"Got mask batch with shape: {masks.shape}")
            except StopIteration as e:
                flush_print(f"Reached end of one of the dataloaders at batch {batch_idx}: {e}")
                break
                
            batch_idx += 1
            flush_print(f"Processing batch {batch_idx} (batch_size: {volumes.shape[0]})")
            
            # Limit batch size if we're near max_samples
            if max_samples:
                remaining_samples = max_samples - samples_processed
                if remaining_samples < volumes.shape[0]:
                    volumes = volumes[:remaining_samples]
                    masks = masks[:remaining_samples]
            
            volumes = volumes.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            flush_print(f"Processing {volumes.shape[0]} volumes with {masks.shape[0]} masks")
            
            # Use mixed precision for faster computation
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                # Use the optimized token extraction from sae_train.py
                tokens = extract_patch_tokens(model, volumes, layer_idx)  # [B*N, D]
                flush_print(f"Extracted {tokens.shape[0]} tokens with {tokens.shape[1]} dimensions")
                
                # Forward through SAE - handle both old and new formats
                if hasattr(sae, 'encode') and callable(getattr(sae, 'encode')):
                    # New LinearSAE format
                    sae_recon, sae_activations = sae(tokens)
                else:
                    # Old format - direct computation
                    if hasattr(sae, 'decoder_bias') and sae.decoder_bias is not None:
                        tokens_centered = tokens - sae.decoder_bias
                        sae_weight = sae.encoder_weight
                        sae_activations = torch.relu(tokens_centered @ sae_weight.T)
                    else:
                        # Fallback to old weight format
                        sae_weight = sae.weight
                        sae_activations = torch.relu(tokens @ sae_weight.T)
                
                flush_print(f"SAE activations shape: {sae_activations.shape}")
            
            # Process masks to get patch-level labels
            B = volumes.shape[0]
            patch_size = 8  # Assuming 8x8x8 patches for 96^3 volumes
            patches_per_dim = 96 // patch_size  # 12
            
            # Reshape masks to patches and take majority vote per patch
            mask_patches = masks.view(B, patches_per_dim, patch_size, 
                                    patches_per_dim, patch_size,
                                    patches_per_dim, patch_size)
            mask_patches = mask_patches.permute(0, 1, 3, 5, 2, 4, 6)  # [B, pd, pd, pd, ps, ps, ps]
            mask_patches = mask_patches.reshape(B, patches_per_dim**3, patch_size**3)
            
            # Take mode (most common value) per patch
            patch_labels = torch.mode(mask_patches, dim=-1)[0]  # [B, N]
            patch_labels = patch_labels.view(-1)  # [B*N]
            
            all_activations.append(sae_activations.cpu())
            all_mask_patches.append(patch_labels.cpu())
            total_tokens += tokens.shape[0]
            samples_processed += volumes.shape[0]
            
            if batch_idx % 10 == 0:  # Log every 10 batches
                flush_print(f"Processed {total_tokens} tokens from {samples_processed} samples so far...")
    
    if not all_activations:
        raise RuntimeError("No activations were extracted! Check mask dataset and volume dataset compatibility.")
    
    final_activations = torch.cat(all_activations, dim=0)
    final_labels = torch.cat(all_mask_patches, dim=0)
    
    flush_print(f"Extraction complete! Total tokens: {final_activations.shape[0]}, SAE dimensions: {final_activations.shape[1]}")
    return final_activations, final_labels


def compute_semantic_correlations(activations, labels, neuron_threshold=0.01):
    """Compute correlations between SAE neurons and semantic labels"""
    n_samples, n_neurons = activations.shape
    n_classes = len(torch.unique(labels))
    
    flush_print(f"Computing correlations for {n_neurons} neurons on {n_samples} patches")
    flush_print(f"Label distribution: {torch.bincount(labels)}")
    
    # Convert labels to one-hot for each semantic class
    semantic_correlations = {}
    semantic_names = ['background', 'membrane', 'sphere', 'cube']
    
    for class_idx, class_name in enumerate(semantic_names):
        if class_idx in labels:
            flush_print(f"Processing {class_name} correlations...")
            class_mask = (labels == class_idx).float()
            correlations = []
            
            for neuron_idx in tqdm(range(n_neurons), desc=f"Computing {class_name} correlations"):
                neuron_acts = activations[:, neuron_idx]
                
                # Only compute correlation if neuron is active enough
                if neuron_acts.mean() > neuron_threshold:
                    corr, p_val = pearsonr(neuron_acts.numpy(), class_mask.numpy())
                    correlations.append((neuron_idx, corr, p_val))
                else:
                    correlations.append((neuron_idx, 0.0, 1.0))
                
                # Progress update every 1000 neurons
                if neuron_idx % 1000 == 0 and neuron_idx > 0:
                    flush_print(f"  Processed {neuron_idx}/{n_neurons} neurons for {class_name}")
            
            semantic_correlations[class_name] = correlations
            flush_print(f"Completed {class_name} correlations")
    
    return semantic_correlations


def visualize_semantic_neurons(correlations, output_dir, top_k=20):
    """Create visualizations of the most semantically selective neurons"""
    flush_print(f"Creating visualizations in {output_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    flush_print("Generating semantic neuron plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SAE Semantic Neuron Analysis', fontsize=16)
    
    # Plot 1: Top correlations per semantic class
    ax = axes[0, 0]
    class_names = list(correlations.keys())
    top_corrs = []
    
    for class_name in class_names:
        corrs = [abs(c[1]) for c in correlations[class_name]]
        top_corrs.append(sorted(corrs, reverse=True)[:top_k])
    
    box_data = []
    labels = []
    for i, class_name in enumerate(class_names):
        box_data.extend(top_corrs[i])
        labels.extend([class_name] * len(top_corrs[i]))
    
    ax.boxplot([top_corrs[i] for i in range(len(class_names))], 
               labels=class_names)
    ax.set_title(f'Distribution of Top {top_k} Correlations')
    ax.set_ylabel('|Correlation|')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 2: Neuron selectivity histogram  
    ax = axes[0, 1]
    all_max_corrs = []
    for neuron_idx in range(len(correlations['background'])):
        max_corr = 0
        for class_name in class_names:
            corr = abs(correlations[class_name][neuron_idx][1])
            max_corr = max(max_corr, corr)
        all_max_corrs.append(max_corr)
    
    ax.hist(all_max_corrs, bins=50, alpha=0.7, edgecolor='black')
    ax.set_title('Neuron Selectivity Distribution')
    ax.set_xlabel('Max |Correlation| across classes')
    ax.set_ylabel('Number of neurons')
    ax.axvline(0.1, color='red', linestyle='--', label='0.1 threshold')
    ax.axvline(0.2, color='orange', linestyle='--', label='0.2 threshold')
    ax.legend()
    
    # Plot 3: Top neurons per class
    ax = axes[1, 0]
    class_colors = ['blue', 'green', 'red', 'orange']
    
    for i, class_name in enumerate(class_names):
        top_neurons = sorted(correlations[class_name], key=lambda x: abs(x[1]), reverse=True)[:10]
        neuron_ids = [n[0] for n in top_neurons]
        corr_vals = [n[1] for n in top_neurons]
        
        y_pos = np.arange(len(neuron_ids)) + i * 0.25
        ax.barh(y_pos, corr_vals, height=0.2, label=class_name, 
                color=class_colors[i], alpha=0.7)
    
    ax.set_title('Top 10 Neurons per Semantic Class')
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Neuron rank')
    ax.legend()
    
    # Plot 4: Correlation matrix between classes
    ax = axes[1, 1]
    n_neurons = len(correlations['background'])
    class_corr_matrix = np.zeros((len(class_names), len(class_names)))
    
    for i, class1 in enumerate(class_names):
        for j, class2 in enumerate(class_names):
            corrs1 = [correlations[class1][n][1] for n in range(n_neurons)]
            corrs2 = [correlations[class2][n][1] for n in range(n_neurons)]
            class_corr_matrix[i, j] = pearsonr(corrs1, corrs2)[0]
    
    sns.heatmap(class_corr_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Correlation between class responses')
    
    plt.tight_layout()
    flush_print("Saving visualization plots...")
    plt.savefig(output_dir / 'semantic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    flush_print("Saving detailed correlation results...")
    results_file = output_dir / 'semantic_correlations.txt'
    with open(results_file, 'w') as f:
        f.write("SAE Semantic Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        for class_name in class_names:
            f.write(f"{class_name.upper()} neurons:\n")
            f.write("-" * 20 + "\n")
            
            # Sort by absolute correlation
            sorted_neurons = sorted(correlations[class_name], 
                                  key=lambda x: abs(x[1]), reverse=True)
            
            f.write("Top 20 correlated neurons:\n")
            for i, (neuron_idx, corr, p_val) in enumerate(sorted_neurons[:20]):
                f.write(f"  {i+1:2d}. Neuron {neuron_idx:4d}: r={corr:6.3f}, p={p_val:.3e}\n")
            f.write("\n")
    
    flush_print(f"Results saved to {output_dir}/")
    
    # Print summary statistics
    flush_print("\nSemantic Analysis Summary:")
    flush_print("=" * 40)
    for class_name in class_names:
        corrs = [abs(c[1]) for c in correlations[class_name]]
        high_corr_count = sum(1 for c in corrs if c > 0.2)
        flush_print(f"{class_name:10s}: {high_corr_count:3d} neurons with |r| > 0.2")
        flush_print(f"              Max correlation: {max(corrs):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAE semantic interpretability")
    parser.add_argument("--sae_checkpoint", type=str, required=True,
                        help="Path to trained SAE checkpoint")
    parser.add_argument("--volume_dir", type=str, required=True,
                        help="Path to volume dataset directory")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Path to mask dataset directory")
    parser.add_argument("--output_dir", type=str, default="sae_semantic_analysis",
                        help="Output directory for results")
    parser.add_argument("--layer", type=int, default=6,
                        help="ViT layer to analyze")
    parser.add_argument("--batch_size", type=int, default=64,  # Increased for H100
                        help="Batch size for data loading")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Maximum number of volumes to analyze")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="bf16",
                        help="Autocast dtype (bf16 recommended for H100)")
    
    args = parser.parse_args()
    
    flush_print("=== SAE Semantic Analysis Starting ===")
    flush_print(f"Arguments: {vars(args)}")
    
    # Setup device with better error handling like inspect_sae.py
    device = get_device()
    flush_print(f"Using device: {device}")
    
    # Set performance optimizations for H100
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        flush_print("Set high precision matmul for better H100 performance")
    
    # Initialize CUDA if available
    if device.type == 'cuda':
        try:
            torch.cuda.init()
            torch.cuda.empty_cache()
            flush_print(f"CUDA initialized successfully")
            flush_print(f"CUDA device count: {torch.cuda.device_count()}")
            flush_print(f"Current CUDA device: {torch.cuda.current_device()}")
            flush_print(f"CUDA device name: {torch.cuda.get_device_name()}")
        except Exception as e:
            flush_print(f"CUDA initialization error: {e}")
            flush_print("Falling back to CPU")
            device = torch.device('cpu')
    
    # Load SAE checkpoint
    flush_print(f"Loading SAE checkpoint from {args.sae_checkpoint}")
    sae_data = load_sae_checkpoint(args.sae_checkpoint)
    
    flush_print(f"SAE info: {sae_data['latent_dim']} latents, layer {sae_data['layer']}")
    flush_print(f"L1 coefficient: {sae_data['l1_coeff']}")
    
    # Load MAE model
    flush_print("Loading MAE model...")
    mae = load_mae_model(sae_data['config'], device)
    
    # Compile MAE for faster inference (H100 optimization)
    flush_print("Compiling MAE model with torch.compile...")
    try:
        mae = torch.compile(mae, backend="inductor", mode="default")
        flush_print("MAE model compiled successfully")
    except Exception as e:
        flush_print(f"Model compilation failed: {e}, continuing without compilation")
    
    # Load SAE - handle both old and new formats
    flush_print("Loading SAE...")
    if sae_data.get('model_state_dict') is not None:
        # New format - full LinearSAE
        sae = LinearSAE(
            input_dim=sae_data['input_dim'],
            latent_dim=sae_data['latent_dim']
        )
        sae.load_state_dict(sae_data['model_state_dict'])
    elif sae_data.get('encoder_weight') is not None and sae_data.get('decoder_weight') is not None:
        # New format but saved as individual tensors
        sae = LinearSAE(
            input_dim=sae_data['input_dim'],
            latent_dim=sae_data['latent_dim']
        )
        # Load individual parameters
        sae.encoder_weight.data = sae_data['encoder_weight']
        sae.decoder_weight.data = sae_data['decoder_weight']
        if sae_data.get('encoder_bias') is not None:
            sae.encoder_bias.data = sae_data['encoder_bias']
        if sae_data.get('decoder_bias') is not None:
            sae.decoder_bias.data = sae_data['decoder_bias']
    else:
        # Old format - create minimal SAE wrapper
        class OldFormatSAE(torch.nn.Module):
            def __init__(self, weight, decoder_bias=None):
                super().__init__()
                if weight is not None:
                    self.register_buffer('weight', weight)
                    self.register_buffer('encoder_weight', weight)
                else:
                    # This shouldn't happen but handle gracefully
                    raise ValueError("No valid SAE weights found in checkpoint")
                if decoder_bias is not None:
                    self.register_buffer('decoder_bias', decoder_bias)
                else:
                    self.decoder_bias = None
            
            def eval(self):
                return self
            
            def train(self, mode=True):
                return self
        
        sae = OldFormatSAE(sae_data['weight'], sae_data['decoder_bias'])
    
    sae = sae.to(device)
    if hasattr(sae, 'eval'):
        sae.eval()
    
    # Compile SAE for faster inference (H100 optimization)
    flush_print("Compiling SAE model with torch.compile...")
    try:
        sae = torch.compile(sae, backend="inductor", mode="default")
        flush_print("SAE model compiled successfully")
    except Exception as e:
        flush_print(f"SAE compilation failed: {e}, continuing without compilation")
    
    # Create datasets
    flush_print("Setting up datasets...")
    
    # Get shard files from volume directory
    volume_shards = sorted(Path(args.volume_dir).glob("shard_*.tar"))
    if not volume_shards:
        raise ValueError(f"No volume shards found in {args.volume_dir}")
    
    flush_print(f"Found {len(volume_shards)} volume shards")
    
    volume_dataset = TarShardDataset(volume_shards, 96, shuffle=False)
    
    mask_dataset = MaskTarDataset(
        mask_dir=Path(args.mask_dir),
        shuffle=False
    )
    
    # Create DataLoaders for both volumes and masks
    flush_print(f"Creating volume dataloader...")
    volume_dataloader = DataLoader(
        volume_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=16,  # Increased for better I/O
        pin_memory=False,  # Dataset already pins memory
        prefetch_factor=4,  # Prefetch for better overlap
        persistent_workers=True,  # Keep workers alive
        drop_last=False,
        timeout=300  # 5 min timeout for large tar files
    )
    
    # Create mask dataloader with same settings
    mask_dataloader = DataLoader(
        mask_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,  # Fewer workers for masks
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=False
    )
    
    flush_print(f"Volume DataLoader created with batch_size={args.batch_size}")
    flush_print(f"Mask DataLoader created with batch_size={args.batch_size}")
    
    # Extract SAE activations
    flush_print("=== Starting SAE activation extraction ===")
    # Use the layer from the SAE checkpoint, not the command line argument
    layer_to_use = sae_data['layer']
    flush_print(f"Using layer {layer_to_use} from SAE checkpoint (overriding --layer {args.layer})")
    
    activations, labels = extract_sae_activations(
        mae, sae, volume_dataloader, device, layer_to_use, args.max_samples,
        use_amp=args.use_amp, amp_dtype=args.amp_dtype, mask_dataloader=mask_dataloader
    )
    
    flush_print(f"=== Activation extraction complete ===")
    flush_print(f"Extracted {activations.shape[0]} patch activations")
    flush_print(f"SAE latent dimension: {activations.shape[1]}")
    
    # Compute semantic correlations
    flush_print("=== Computing semantic correlations ===")
    correlations = compute_semantic_correlations(activations, labels)
    
    # Visualize results
    flush_print("=== Creating visualizations ===")
    visualize_semantic_neurons(correlations, args.output_dir)
    
    flush_print("=== Semantic analysis complete! ===")


if __name__ == "__main__":
    main() 