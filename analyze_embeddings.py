#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Analysis for Spatial Invariance
==========================================
Analyzes whether patch embeddings are location-invariant (semantic) or 
position-dependent for base vs base_patch_conv architectures.

Tests:
1. Pairwise similarity: sphere-sphere vs random-random patches
2. Linear probes: semantic vs coordinate prediction
3. Representational Similarity Analysis (RSA)
4. Translation sensitivity index
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Import our models and generator
from vit_3d import mae_vit_3d_base, mae_vit_3d_base_patch_conv, get_device
from vol_generator import MembraneGen
from vol_train import load_model_checkpoint

try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('default')

class EmbeddingAnalyzer:
    """Analyzes spatial invariance in ViT patch embeddings."""
    
    def __init__(self, model, model_name, device):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.model.eval()
        
        # Get patch embedding function
        self.patch_embed = model.encoder.patch_embed
        self.pos_embed = model.encoder.pos_embed[:, 1:, :]  # Skip CLS token
        
        # Store results
        self.results = {}
    
    def extract_patch_embeddings(self, volume, add_pos_embed=False):
        """Extract patch embeddings from a volume.
        
        Args:
            volume: (1, 1, D, H, W) tensor
            add_pos_embed: Whether to add positional embeddings
            
        Returns:
            embeddings: (L, embed_dim) tensor where L = num_patches
        """
        with torch.no_grad():
            # Get patch embeddings
            x = self.patch_embed(volume)  # (1, L, embed_dim)
            
            if add_pos_embed:
                x = x + self.pos_embed
                
            return x.squeeze(0)  # (L, embed_dim)
    
    def get_patch_masks(self, volume_mask, patch_size=8):
        """Get semantic label for each patch based on majority vote.
        
        Args:
            volume_mask: (D, H, W) segmentation mask
            patch_size: Size of patches
            
        Returns:
            patch_labels: (L,) array of semantic labels per patch
            patch_coords: (L, 3) array of (d, h, w) patch coordinates
        """
        D, H, W = volume_mask.shape
        pd, ph, pw = D // patch_size, H // patch_size, W // patch_size
        
        patch_labels = []
        patch_coords = []
        
        for d in range(pd):
            for h in range(ph):
                for w in range(pw):
                    # Extract patch
                    patch = volume_mask[
                        d*patch_size:(d+1)*patch_size,
                        h*patch_size:(h+1)*patch_size,
                        w*patch_size:(w+1)*patch_size
                    ]
                    
                    # Majority vote for semantic label
                    unique, counts = np.unique(patch, return_counts=True)
                    majority_label = unique[np.argmax(counts)]
                    
                    patch_labels.append(majority_label)
                    patch_coords.append([d, h, w])
        
        return np.array(patch_labels), np.array(patch_coords)
    
    def test_pairwise_similarity(self, volumes, masks, num_pairs=1000):
        """Test 1: Pairwise similarity analysis."""
        print(f"Running pairwise similarity test for {self.model_name}...")
        
        # Collect embeddings and labels
        all_embeddings_raw = []
        all_embeddings_pos = []
        all_labels = []
        all_coords = []
        
        for vol, mask in tqdm(zip(volumes, masks), desc="Processing volumes"):
            vol_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Get embeddings with and without positional encoding
            emb_raw = self.extract_patch_embeddings(vol_tensor, add_pos_embed=False)
            emb_pos = self.extract_patch_embeddings(vol_tensor, add_pos_embed=True)
            
            # Get patch labels
            patch_labels, patch_coords = self.get_patch_masks(mask)
            
            all_embeddings_raw.append(emb_raw.float().cpu().numpy())
            all_embeddings_pos.append(emb_pos.float().cpu().numpy())
            all_labels.append(patch_labels)
            all_coords.append(patch_coords)
        
        # Flatten everything
        embeddings_raw = np.vstack(all_embeddings_raw)
        embeddings_pos = np.vstack(all_embeddings_pos)
        labels = np.concatenate(all_labels)
        coords = np.vstack(all_coords)
        
        # Find sphere patches (label == 2)
        sphere_indices = np.where(labels == 2)[0]
        
        similarities = {}
        
        for emb_type, embeddings in [("raw", embeddings_raw), ("with_pos", embeddings_pos)]:
            sphere_sphere_sims = []
            random_random_sims = []
            
            # Sample pairs
            for _ in range(min(num_pairs, len(sphere_indices) // 2)):
                # Sphere-sphere pairs
                if len(sphere_indices) >= 2:
                    idx1, idx2 = np.random.choice(sphere_indices, 2, replace=False)
                    sim = F.cosine_similarity(
                        torch.from_numpy(embeddings[idx1]).unsqueeze(0),
                        torch.from_numpy(embeddings[idx2]).unsqueeze(0)
                    ).item()
                    sphere_sphere_sims.append(sim)
                
                # Random-random pairs
                idx1, idx2 = np.random.choice(len(embeddings), 2, replace=False)
                sim = F.cosine_similarity(
                    torch.from_numpy(embeddings[idx1]).unsqueeze(0),
                    torch.from_numpy(embeddings[idx2]).unsqueeze(0)
                ).item()
                random_random_sims.append(sim)
            
            similarities[emb_type] = {
                'sphere_sphere': sphere_sphere_sims,
                'random_random': random_random_sims
            }
        
        self.results['pairwise_similarity'] = similarities
        return similarities
    
    def test_linear_probes(self, volumes, masks, test_size=0.3):
        """Test 2: Linear probe analysis."""
        print(f"Running linear probe test for {self.model_name}...")
        
        # Collect data
        embeddings_raw = []
        embeddings_pos = []
        semantic_labels = []
        coordinate_labels = []
        
        for vol, mask in tqdm(zip(volumes, masks), desc="Collecting probe data"):
            vol_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(self.device)
            
            emb_raw = self.extract_patch_embeddings(vol_tensor, add_pos_embed=False)
            emb_pos = self.extract_patch_embeddings(vol_tensor, add_pos_embed=True)
            
            patch_labels, patch_coords = self.get_patch_masks(mask)
            
            embeddings_raw.append(emb_raw.float().cpu().numpy())
            embeddings_pos.append(emb_pos.float().cpu().numpy())
            semantic_labels.append(patch_labels)
            
            # Convert coordinates to single labels for coordinate probe
            coord_labels = patch_coords[:, 0] * 144 + patch_coords[:, 1] * 12 + patch_coords[:, 2]  # 12x12x12 grid
            coordinate_labels.append(coord_labels)
        
        # Flatten
        embeddings_raw = np.vstack(embeddings_raw)
        embeddings_pos = np.vstack(embeddings_pos)
        semantic_labels = np.concatenate(semantic_labels)
        coordinate_labels = np.concatenate(coordinate_labels)
        
        # Sample subset for faster training
        n_samples = min(20000, len(embeddings_raw))
        indices = np.random.choice(len(embeddings_raw), n_samples, replace=False)
        
        probe_results = {}
        
        for emb_type, embeddings in [("raw", embeddings_raw), ("with_pos", embeddings_pos)]:
            X = embeddings[indices]
            y_semantic = semantic_labels[indices]
            y_coordinate = coordinate_labels[indices]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            split_idx = int(len(X_scaled) * (1 - test_size))
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_sem_train, y_sem_test = y_semantic[:split_idx], y_semantic[split_idx:]
            y_coord_train, y_coord_test = y_coordinate[:split_idx], y_coordinate[split_idx:]
            
            # Semantic probe
            semantic_probe = LogisticRegression(max_iter=1000, random_state=42)
            semantic_probe.fit(X_train, y_sem_train)
            sem_acc = accuracy_score(y_sem_test, semantic_probe.predict(X_test))
            
            # Coordinate probe
            coordinate_probe = LogisticRegression(max_iter=1000, random_state=42)
            coordinate_probe.fit(X_train, y_coord_train)
            coord_acc = accuracy_score(y_coord_test, coordinate_probe.predict(X_test))
            
            probe_results[emb_type] = {
                'semantic_accuracy': sem_acc,
                'coordinate_accuracy': coord_acc
            }
        
        self.results['linear_probes'] = probe_results
        return probe_results
    
    def test_translation_sensitivity(self, volumes, masks, num_tests=100):
        """Test 4: Translation sensitivity index."""
        print(f"Running translation sensitivity test for {self.model_name}...")
        
        sensitivity_scores = []
        
        for _ in tqdm(range(num_tests), desc="Testing translations"):
            # Pick a random volume
            vol_idx = np.random.randint(len(volumes))
            vol = volumes[vol_idx]
            mask = masks[vol_idx]
            
            # Find a sphere patch
            patch_labels, patch_coords = self.get_patch_masks(mask)
            sphere_indices = np.where(patch_labels == 2)[0]
            
            if len(sphere_indices) == 0:
                continue
                
            # Pick a random sphere patch
            patch_idx = np.random.choice(sphere_indices)
            d, h, w = patch_coords[patch_idx]
            
            # Get original embedding
            vol_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(self.device)
            orig_emb = self.extract_patch_embeddings(vol_tensor, add_pos_embed=False)
            orig_patch_emb = orig_emb[patch_idx].float().cpu().numpy()
            
            # Apply random translation (circular shift)
            shift_d = np.random.randint(-20, 21)
            shift_h = np.random.randint(-20, 21)
            shift_w = np.random.randint(-20, 21)
            
            vol_shifted = np.roll(vol, (shift_d, shift_h, shift_w), axis=(0, 1, 2))
            vol_shifted_tensor = torch.from_numpy(vol_shifted).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Get shifted embedding at same patch location
            shifted_emb = self.extract_patch_embeddings(vol_shifted_tensor, add_pos_embed=False)
            shifted_patch_emb = shifted_emb[patch_idx].float().cpu().numpy()
            
            # Compute embedding difference
            diff = np.linalg.norm(orig_patch_emb - shifted_patch_emb)
            sensitivity_scores.append(diff)
        
        avg_sensitivity = np.mean(sensitivity_scores)
        self.results['translation_sensitivity'] = {
            'scores': sensitivity_scores,
            'average': avg_sensitivity
        }
        
        return avg_sensitivity
    
    def visualize_results(self, save_dir):
        """Create visualizations of all results."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # 1. Pairwise similarity histograms
        if 'pairwise_similarity' in self.results:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Pairwise Similarity Analysis - {self.model_name}', fontsize=16)
            
            for i, emb_type in enumerate(['raw', 'with_pos']):
                data = self.results['pairwise_similarity'][emb_type]
                
                # Plot histograms
                axes[i, 0].hist(data['sphere_sphere'], alpha=0.7, label='Sphere-Sphere', bins=30)
                axes[i, 0].hist(data['random_random'], alpha=0.7, label='Random-Random', bins=30)
                axes[i, 0].set_title(f'Embeddings: {emb_type}')
                axes[i, 0].set_xlabel('Cosine Similarity')
                axes[i, 0].set_ylabel('Count')
                axes[i, 0].legend()
                
                # Plot means
                means = [np.mean(data['sphere_sphere']), np.mean(data['random_random'])]
                axes[i, 1].bar(['Sphere-Sphere', 'Random-Random'], means)
                axes[i, 1].set_title(f'Mean Similarity: {emb_type}')
                axes[i, 1].set_ylabel('Mean Cosine Similarity')
            
            plt.tight_layout()
            plt.savefig(save_dir / f'pairwise_similarity_{self.model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Linear probe results
        if 'linear_probes' in self.results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            probe_data = self.results['linear_probes']
            x = np.arange(2)
            width = 0.35
            
            raw_scores = [probe_data['raw']['semantic_accuracy'], probe_data['raw']['coordinate_accuracy']]
            pos_scores = [probe_data['with_pos']['semantic_accuracy'], probe_data['with_pos']['coordinate_accuracy']]
            
            ax.bar(x - width/2, raw_scores, width, label='Raw Embeddings', alpha=0.8)
            ax.bar(x + width/2, pos_scores, width, label='With Position Encoding', alpha=0.8)
            
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Linear Probe Accuracy - {self.model_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(['Semantic', 'Coordinate'])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / f'linear_probes_{self.model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

def generate_test_data(num_volumes=50, volume_size=96):
    """Generate test volumes with known semantic structure."""
    print(f"Generating {num_volumes} test volumes...")
    
    generator = MembraneGen(generate_masks=True, equal_combinations=True)
    
    volumes = []
    masks = []
    
    for i in tqdm(range(num_volumes), desc="Generating volumes"):
        vol_bytes, mask_bytes = generator(i)
        
        # Convert bytes to numpy arrays
        vol = np.frombuffer(vol_bytes, dtype=np.float32).reshape(volume_size, volume_size, volume_size)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(volume_size, volume_size, volume_size)
        
        volumes.append(vol)
        masks.append(mask)
    
    return volumes, masks

def load_models(checkpoint_dir, device):
    """Load both base and base_patch_conv models."""
    models = {}
    
    # Try to load checkpoints
    checkpoint_dir = Path(checkpoint_dir)
    
    for model_name in ['base', 'base_patch_conv']:
        print(f"Loading {model_name} model...")
        
        # Create model with correct settings (matching training)
        if model_name == 'base':
            model = mae_vit_3d_base(volume_size=(96, 96, 96), patch_size=8)
        else:
            model = mae_vit_3d_base_patch_conv(volume_size=(96, 96, 96), patch_size=8)
        
        # Apply memory format optimization (matching training setup)
        try:
            model = model.to(device, memory_format=torch.channels_last_3d)
            print(f"  Using channels-last-3D memory format for {model_name}")
        except (AttributeError, RuntimeError):
            model = model.to(device)
            print(f"  Using default memory format for {model_name}")
        
        # Look for checkpoint files (try multiple patterns)
        if model_name == 'base':
            checkpoint_patterns = [
                "best_model_mask_75_v3_base.pt",  # Specific base model
                "best_model_mask_75_v2_base.pt",  # Alternative base model
                "best_model_*base*.pt",           # Any base model
                "best_model_*.pt"                 # Fallback
            ]
        else:  # base_patch_conv
            checkpoint_patterns = [
                f"best_model_*{model_name}*.pt",
                f"*{model_name}*.pt",
                f"best_model_*.pt"  # Fallback to any best model
            ]
        
        checkpoint_files = []
        for pattern in checkpoint_patterns:
            checkpoint_files = list(checkpoint_dir.glob(pattern))
            if checkpoint_files:
                break
        
        if checkpoint_files:
            # Sort by modification time and take the most recent
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            checkpoint_path = checkpoint_files[0]
            print(f"  Found checkpoint: {checkpoint_path}")
            
            try:
                model, checkpoint = load_model_checkpoint(checkpoint_path, model, device)
                
                # Print checkpoint info
                if 'global_step' in checkpoint:
                    print(f"  Checkpoint from step {checkpoint['global_step']}")
                if 'val_loss' in checkpoint:
                    print(f"  Best validation loss: {checkpoint['val_loss']:.6f}")
                if 'model_dtype' in checkpoint:
                    print(f"  Training dtype: {checkpoint['model_dtype']}")
                    
            except Exception as e:
                print(f"  Warning: Failed to load checkpoint {checkpoint_path}: {e}")
                print(f"  Using random weights for {model_name}")
        else:
            print(f"  No checkpoint found for {model_name}, using random weights")
        
        # Ensure model is in eval mode for analysis
        model.eval()
        
        # Disable compilation for analysis (since we need to access internals)
        # Note: Training uses torch.compile but we need uncompiled model for embedding extraction
        if hasattr(model, '_orig_mod'):
            # If it's a compiled model, get the original
            model = model._orig_mod
        
        models[model_name] = model
    
    return models

def main(args):
    device = get_device()
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS FOR SPATIAL INVARIANCE")
    print("="*80)
    print("This analysis compares 'base' vs 'base_patch_conv' architectures:")
    print("• base: Standard 3D patch embedding (spatial position encoded)")
    print("• base_patch_conv: Convolutional patch embedding (translation-invariant)")
    print()
    print("Key training details:")
    print("• Models trained with bf16 autocast but saved in fp32")
    print("• torch.compile used during training (disabled for analysis)")
    print("• channels_last_3d memory format for conv optimization")
    print("• EMA disabled in current training setup")
    print("="*80 + "\n")
    
    # Generate or load test data
    if args.data_path:
        print(f"Loading test data from {args.data_path}")
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)
        volumes, masks = data['volumes'], data['masks']
    else:
        volumes, masks = generate_test_data(args.num_volumes, args.volume_size)
        
        # Save for reuse
        if args.save_data:
            data_path = Path(args.output_dir) / 'test_data.pkl'
            with open(data_path, 'wb') as f:
                pickle.dump({'volumes': volumes, 'masks': masks}, f)
            print(f"Saved test data to {data_path}")
    
    # Load models
    models = load_models(args.checkpoint_dir, device)
    
    # Run analysis for each model
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {model_name} model")
        print(f"{'='*60}")
        
        analyzer = EmbeddingAnalyzer(model, model_name, device)
        
        # Run tests
        if args.test_similarity:
            analyzer.test_pairwise_similarity(volumes, masks, args.num_pairs)
        
        if args.test_probes:
            analyzer.test_linear_probes(volumes, masks)
        
        if args.test_translation:
            analyzer.test_translation_sensitivity(volumes, masks, args.num_translations)
        
        # Save results
        all_results[model_name] = analyzer.results
        
        # Create visualizations
        analyzer.visualize_results(args.output_dir)
    
    # Save all results
    results_path = Path(args.output_dir) / 'analysis_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_numpy(all_results), f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for model_name in models.keys():
        results = all_results[model_name]
        print(f"\n{model_name.upper()} MODEL:")
        
        if 'linear_probes' in results:
            probes = results['linear_probes']
            print(f"  Linear Probes (raw embeddings):")
            print(f"    Semantic accuracy: {probes['raw']['semantic_accuracy']:.3f}")
            print(f"    Coordinate accuracy: {probes['raw']['coordinate_accuracy']:.3f}")
            print(f"  Linear Probes (with position):")
            print(f"    Semantic accuracy: {probes['with_pos']['semantic_accuracy']:.3f}")
            print(f"    Coordinate accuracy: {probes['with_pos']['coordinate_accuracy']:.3f}")
        
        if 'pairwise_similarity' in results:
            sim = results['pairwise_similarity']
            print(f"  Pairwise Similarity (raw embeddings):")
            print(f"    Sphere-sphere mean: {np.mean(sim['raw']['sphere_sphere']):.3f}")
            print(f"    Random-random mean: {np.mean(sim['raw']['random_random']):.3f}")
        
        if 'translation_sensitivity' in results:
            trans = results['translation_sensitivity']
            print(f"  Translation Sensitivity: {trans['average']:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze spatial invariance in ViT embeddings")
    
    # Data arguments
    parser.add_argument("--output_dir", type=str, default="analysis_results", 
                        help="Directory to save results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to pre-generated test data (pickle file)")
    parser.add_argument("--save_data", action="store_true",
                        help="Save generated test data for reuse")
    
    # Generation arguments
    parser.add_argument("--num_volumes", type=int, default=50,
                        help="Number of test volumes to generate")
    parser.add_argument("--volume_size", type=int, default=96,
                        help="Size of test volumes")
    
    # Test arguments
    parser.add_argument("--test_similarity", action="store_true", default=True,
                        help="Run pairwise similarity test")
    parser.add_argument("--test_probes", action="store_true", default=True,
                        help="Run linear probe tests")
    parser.add_argument("--test_translation", action="store_true", default=True,
                        help="Run translation sensitivity test")
    
    # Test parameters
    parser.add_argument("--num_pairs", type=int, default=1000,
                        help="Number of pairs for similarity test")
    parser.add_argument("--num_translations", type=int, default=100,
                        help="Number of translations for sensitivity test")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    main(args) 