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

from util_files.config import VitConfig
from vit_3d import VisionTransformer3D
from sae_train import SAE

warnings.filterwarnings("ignore", category=UserWarning)


class MaskTarDataset:
    """Dataset for loading segmentation masks from tar files"""
    
    def __init__(self, mask_dir: Path, shard_pattern="shard_*.tar"):
        self.mask_files = sorted(mask_dir.glob(shard_pattern))
        if not self.mask_files:
            raise ValueError(f"No mask files found in {mask_dir}")
        
        # Count total masks
        self.total_masks = 0
        self.shard_sizes = []
        for mask_file in self.mask_files:
            with tarfile.open(mask_file, 'r') as tar:
                members = [m for m in tar.getmembers() if m.name.endswith('.bin')]
                self.shard_sizes.append(len(members))
                self.total_masks += len(members)
        
        print(f"Found {len(self.mask_files)} mask shards with {self.total_masks} total masks")
    
    def __len__(self):
        return self.total_masks
    
    def __getitem__(self, idx):
        # Find which shard contains this index
        shard_idx = 0
        remaining_idx = idx
        for i, shard_size in enumerate(self.shard_sizes):
            if remaining_idx < shard_size:
                shard_idx = i
                break
            remaining_idx -= shard_size
        
        # Load from the appropriate shard
        with tarfile.open(self.mask_files[shard_idx], 'r') as tar:
            members = sorted([m for m in tar.getmembers() if m.name.endswith('.bin')])
            mask_member = members[remaining_idx]
            mask_data = tar.extractfile(mask_member).read()
            mask = np.frombuffer(mask_data, dtype=np.uint8).reshape(96, 96, 96)
        
        return torch.from_numpy(mask)


def extract_sae_activations(model, sae, dataloader, device, layer_idx, max_samples=None):
    """Extract SAE activations for all patches in the dataset"""
    model.eval()
    sae.eval()
    
    all_activations = []
    all_mask_patches = []
    
    with torch.no_grad():
        for batch_idx, (volumes, masks) in enumerate(tqdm(dataloader, desc="Extracting activations")):
            if max_samples and batch_idx * dataloader.batch_size >= max_samples:
                break
                
            volumes = volumes.to(device)
            masks = masks.to(device)
            
            # Forward through ViT to get intermediate activations
            x = model.patch_embed(volumes)
            B, N, D = x.shape  # batch_size, num_patches, embed_dim
            
            # Add position embeddings and process through transformer
            if hasattr(model, 'pos_embed') and model.pos_embed is not None:
                x = x + model.pos_embed[:, :N]
            
            for i, block in enumerate(model.blocks):
                x = block(x)
                if i == layer_idx:
                    target_activations = x.clone()
                    break
            
            # Reshape activations to get per-patch features
            # target_activations: [B, N, D] where N = num_patches
            target_activations = target_activations.view(-1, target_activations.size(-1))  # [B*N, D]
            
            # Forward through SAE
            sae_activations = sae(target_activations)  # [B*N, latent_dim]
            
            # Process masks to get patch-level labels
            # masks: [B, 96, 96, 96], need to convert to [B, N] patch labels
            patch_size = model.patch_embed.patch_size[0]  # assuming cubic patches
            patches_per_dim = 96 // patch_size
            
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
    
    return torch.cat(all_activations, dim=0), torch.cat(all_mask_patches, dim=0)


def compute_semantic_correlations(activations, labels, neuron_threshold=0.01):
    """Compute correlations between SAE neurons and semantic labels"""
    n_samples, n_neurons = activations.shape
    n_classes = len(torch.unique(labels))
    
    print(f"Computing correlations for {n_neurons} neurons on {n_samples} patches")
    print(f"Label distribution: {torch.bincount(labels)}")
    
    # Convert labels to one-hot for each semantic class
    semantic_correlations = {}
    semantic_names = ['background', 'membrane', 'sphere', 'cube']
    
    for class_idx, class_name in enumerate(semantic_names):
        if class_idx in labels:
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
            
            semantic_correlations[class_name] = correlations
    
    return semantic_correlations


def visualize_semantic_neurons(correlations, output_dir, top_k=20):
    """Create visualizations of the most semantically selective neurons"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    plt.savefig(output_dir / 'semantic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
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
    
    print(f"Results saved to {output_dir}/")
    
    # Print summary statistics
    print("\nSemantic Analysis Summary:")
    print("=" * 40)
    for class_name in class_names:
        corrs = [abs(c[1]) for c in correlations[class_name]]
        high_corr_count = sum(1 for c in corrs if c > 0.2)
        print(f"{class_name:10s}: {high_corr_count:3d} neurons with |r| > 0.2")
        print(f"              Max correlation: {max(corrs):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAE semantic interpretability")
    parser.add_argument("--sae_checkpoint", type=str, required=True,
                        help="Path to trained SAE checkpoint")
    parser.add_argument("--vit_checkpoint", type=str, required=True,
                        help="Path to trained ViT checkpoint")
    parser.add_argument("--volume_dir", type=str, required=True,
                        help="Path to volume dataset directory")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Path to mask dataset directory")
    parser.add_argument("--output_dir", type=str, default="sae_semantic_analysis",
                        help="Output directory for results")
    parser.add_argument("--layer", type=int, default=6,
                        help="ViT layer to analyze")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for data loading")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum number of volumes to analyze")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load ViT model
    print("Loading ViT model...")
    config = VitConfig()
    model = VisionTransformer3D(config)
    checkpoint = torch.load(args.vit_checkpoint, map_location=device)
    
    # Handle potential torch.compile prefixes
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key.replace('_orig_mod.', '')
        cleaned_state_dict[clean_key] = value
    
    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    
    # Load SAE
    print("Loading SAE...")
    sae_checkpoint = torch.load(args.sae_checkpoint, map_location=device)
    sae = SAE(
        input_dim=sae_checkpoint['input_dim'],
        latent_dim=sae_checkpoint['latent_dim']
    )
    sae.load_state_dict(sae_checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()
    
    # Create datasets - we need to create a combined dataset that loads both volumes and masks
    print("Setting up datasets...")
    
    # For now, let's create a simple approach that loads masks separately
    # In practice, you'd want to modify the TarShardDataset to also load masks
    
    from util_files.train_helper import TarShardDataset
    
    volume_dataset = TarShardDataset(
        data_dir=args.volume_dir,
        shard_pattern="shard_*.tar",
        img_size=96,
        augment=False
    )
    
    mask_dataset = MaskTarDataset(
        mask_dir=Path(args.mask_dir)
    )
    
    # Create combined dataset
    combined_dataset = list(zip(volume_dataset, mask_dataset))
    dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=4)
    
    # Extract SAE activations
    print("Extracting SAE activations...")
    activations, labels = extract_sae_activations(
        model, sae, dataloader, device, args.layer, args.max_samples
    )
    
    print(f"Extracted {activations.shape[0]} patch activations")
    print(f"SAE latent dimension: {activations.shape[1]}")
    
    # Compute semantic correlations
    print("Computing semantic correlations...")
    correlations = compute_semantic_correlations(activations, labels)
    
    # Visualize results
    print("Creating visualizations...")
    visualize_semantic_neurons(correlations, args.output_dir)
    
    print("Semantic analysis complete!")


if __name__ == "__main__":
    main() 