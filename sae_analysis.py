#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive SAE Monosemanticity Analysis for 3D ViT-MAE
=========================================================

This script analyzes the Sparse Autoencoder (SAE) trained on 3D ViT-MAE activations
to identify monosemantic neurons and evaluate reconstruction quality.

Key analyses:
1. Monosemantic neuron extraction using feature activation patterns
2. Reconstruction quality comparison (Input vs ViT vs ViT+SAE)
3. Interactive visualization with volume sliders
4. Feature attribution analysis inspired by Anthropic's work
5. Dead neuron analysis and feature statistics

Expected monosemantic features for synthetic EM data:
- Background detectors (low activation uniform regions)
- Membrane detectors (thin sheet structures)
- Sphere detectors (spherical objects)
- Cube detectors (cubic/angular objects)
- Edge detectors (boundaries between objects)
- Texture detectors (surface patterns)
"""

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tarfile
import io

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Visualization and analysis
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

# Project imports
from vol_train import TarShardDataset, CUDAPrefetcher
from sae_train import LinearSAE, GatedSAE, TokenExtractor
from vit_3d import (
    mae_vit_3d_base_conv, mae_vit_3d_small_conv, mae_vit_3d_large_conv,
    mae_vit_3d_hemibrain_optimal_conv, mae_vit_3d_base_patch_conv, get_device
)

# Set up plotting
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# AMP dtype mapping
_AMP_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

class SAEAnalyzer:
    """Comprehensive SAE analysis toolkit for monosemanticity detection."""
    
    def __init__(self, vit_model, sae_model, token_extractor, act_mean, act_std, device):
        self.vit = vit_model
        self.sae = sae_model
        self.token_extractor = token_extractor
        self.act_mean = act_mean
        self.act_std = act_std
        self.device = device
        
        # Cache for feature analysis
        self.feature_cache = {}
        self.monosemantic_neurons = {}
        
        # Object type mapping for synthetic data
        self.object_types = {
            0: "background",
            1: "membrane", 
            2: "sphere",
            3: "cube"
        }
        
        print(f"SAE Analyzer initialized:")
        print(f"  - SAE latent dim: {self.sae.encoder_weight.shape[0]}")
        print(f"  - Input feature dim: {self.sae.encoder_weight.shape[1]}")
        print(f"  - Device: {device}")
    
    @torch.no_grad()
    def extract_features_and_activations(self, volumes, masks=None):
        """Extract VIT features and SAE activations for analysis."""
        # Get VIT features
        vit_tokens = self.token_extractor.extract_tokens(volumes)
        vit_tokens_wh = (vit_tokens - self.act_mean) / self.act_std
        
        # Get SAE activations
        if hasattr(self.sae, 'encode'):
            sae_activations = self.sae.encode(vit_tokens_wh)
        else:
            sae_recon, sae_activations = self.sae(vit_tokens_wh)
        
        # Get reconstructions
        sae_recon = self.sae.decode(sae_activations)
        
        return {
            'vit_tokens': vit_tokens,
            'vit_tokens_wh': vit_tokens_wh,
            'sae_activations': sae_activations,
            'sae_reconstruction': sae_recon,
            'masks': masks
        }
    
    def analyze_neuron_selectivity(self, volumes, masks, max_batches=50):
        """Analyze which SAE neurons are selective to different object types."""
        print("Analyzing neuron selectivity to object types...")
        
        all_activations = []
        all_object_labels = []
        
        batch_count = 0
        for vol_batch, mask_batch in zip(volumes, masks):
            if batch_count >= max_batches:
                break
                
            vol_batch = vol_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)
            
            # Extract features
            data = self.extract_features_and_activations(vol_batch, mask_batch)
            activations = data['sae_activations']  # (batch_size * num_patches, latent_dim)
            
            # Reshape volumes and masks to match token structure
            B, C, D, H, W = vol_batch.shape
            patch_size = 8  # From your config
            patches_per_dim = D // patch_size
            
            # Reshape masks to match patch layout
            mask_patches = F.avg_pool3d(
                mask_batch.float(), 
                kernel_size=patch_size, 
                stride=patch_size
            )
            mask_patches = mask_patches.view(B, -1).long()  # (B, num_patches)
            
            # Get dominant object type per patch (mode)
            patch_labels = []
            for b in range(B):
                for p in range(mask_patches.shape[1]):
                    # Get the most common object type in this patch
                    z = p // (patches_per_dim ** 2)
                    y = (p % (patches_per_dim ** 2)) // patches_per_dim
                    x = p % patches_per_dim
                    
                    # Extract patch region from original mask
                    patch_mask = mask_batch[b, 0, 
                                          z*patch_size:(z+1)*patch_size,
                                          y*patch_size:(y+1)*patch_size,
                                          x*patch_size:(x+1)*patch_size]
                    
                    # Get mode (most common value)
                    unique_vals, counts = torch.unique(patch_mask, return_counts=True)
                    dominant_type = unique_vals[torch.argmax(counts)].item()
                    patch_labels.append(dominant_type)
            
            patch_labels = torch.tensor(patch_labels, device=self.device)
            
            all_activations.append(activations.cpu())
            all_object_labels.append(patch_labels.cpu())
            
            batch_count += 1
        
        # Concatenate all data
        all_activations = torch.cat(all_activations, dim=0)  # (total_patches, latent_dim)
        all_object_labels = torch.cat(all_object_labels, dim=0)  # (total_patches,)
        
        print(f"Analyzing {all_activations.shape[0]} patches across {len(self.object_types)} object types")
        
        # Analyze selectivity for each neuron
        selectivity_results = {}
        
        for neuron_idx in tqdm(range(all_activations.shape[1]), desc="Analyzing neurons"):
            neuron_acts = all_activations[:, neuron_idx]
            
            # Skip completely dead neurons
            if neuron_acts.max() < 1e-6:
                continue
                
            # Calculate selectivity metrics
            selectivity = self._calculate_selectivity_metrics(neuron_acts, all_object_labels)
            selectivity_results[neuron_idx] = selectivity
        
        return selectivity_results
    
    def _calculate_selectivity_metrics(self, activations, labels):
        """Calculate various selectivity metrics for a neuron."""
        # Only consider patches where neuron is active
        active_threshold = activations.quantile(0.95)  # Top 5% activations
        active_mask = activations > active_threshold
        
        if active_mask.sum() < 10:  # Need at least 10 active patches
            return None
        
        active_labels = labels[active_mask]
        
        # Calculate metrics
        total_active = active_mask.sum().item()
        
        # Object type distribution in active patches
        object_dist = {}
        for obj_type in range(4):
            count = (active_labels == obj_type).sum().item()
            object_dist[obj_type] = count / total_active
        
        # Selectivity index (entropy-based)
        probs = torch.tensor(list(object_dist.values()))
        probs = probs[probs > 0]  # Remove zeros
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        max_entropy = torch.log(torch.tensor(4.0))  # log(4) for 4 object types
        selectivity_index = 1 - (entropy / max_entropy)
        
        # Dominant object type
        dominant_type = max(object_dist, key=object_dist.get)
        dominance_score = object_dist[dominant_type]
        
        return {
            'selectivity_index': selectivity_index.item(),
            'dominance_score': dominance_score,
            'dominant_type': dominant_type,
            'object_distribution': object_dist,
            'total_active_patches': total_active,
            'activation_threshold': active_threshold.item(),
            'mean_activation': activations.mean().item(),
            'max_activation': activations.max().item(),
            'sparsity': (activations > 1e-6).float().mean().item()
        }
    
    def find_monosemantic_neurons(self, selectivity_results, min_selectivity=0.7, min_dominance=0.6):
        """Identify monosemantic neurons based on selectivity analysis."""
        print(f"Finding monosemantic neurons (selectivity > {min_selectivity}, dominance > {min_dominance})")
        
        monosemantic = {}
        
        for neuron_idx, metrics in selectivity_results.items():
            if metrics is None:
                continue
                
            if (metrics['selectivity_index'] > min_selectivity and 
                metrics['dominance_score'] > min_dominance):
                
                obj_type = metrics['dominant_type']
                obj_name = self.object_types[obj_type]
                
                if obj_name not in monosemantic:
                    monosemantic[obj_name] = []
                
                monosemantic[obj_name].append({
                    'neuron_idx': neuron_idx,
                    'selectivity': metrics['selectivity_index'],
                    'dominance': metrics['dominance_score'],
                    'total_active': metrics['total_active_patches'],
                    'metrics': metrics
                })
        
        # Sort by selectivity within each object type
        for obj_name in monosemantic:
            monosemantic[obj_name].sort(key=lambda x: x['selectivity'], reverse=True)
        
        self.monosemantic_neurons = monosemantic
        
        # Print summary
        print("\nMonosemantic Neurons Found:")
        for obj_name, neurons in monosemantic.items():
            print(f"  {obj_name}: {len(neurons)} neurons")
            for i, neuron in enumerate(neurons[:3]):  # Show top 3
                print(f"    #{i+1}: Neuron {neuron['neuron_idx']} "
                      f"(selectivity: {neuron['selectivity']:.3f}, "
                      f"dominance: {neuron['dominance']:.3f})")
        
        return monosemantic
    
    def analyze_reconstruction_quality(self, volumes, masks, num_samples=10):
        """Analyze reconstruction quality through VIT and SAE."""
        print("Analyzing reconstruction quality...")
        
        results = []
        
        for i in range(min(num_samples, len(volumes))):
            vol = volumes[i].to(self.device)  # Get individual tensor from list
            mask = masks[i].to(self.device) if masks is not None else None
            
            # Use the same masking ratio as MAE training (75 %) so that only
            # masked patches are reconstructed – visible-patch predictions were
            # never in the loss and are effectively noise.
            MAE_MASK_RATIO = 0.75

            # Get VIT reconstruction (through MAE)
            with torch.no_grad():
                vit_loss, vit_pred, vit_mask, patch_stats = self.vit(vol, mask_ratio=MAE_MASK_RATIO)
                
                # Convert VIT predictions back to volume format
                vit_recon = self.vit.unpatchify(vit_pred)
                
                # Get SAE reconstruction
                data = self.extract_features_and_activations(vol, mask)
                sae_recon_tokens = data['sae_reconstruction']
                
                # Convert SAE tokens back to original space
                sae_recon_tokens_orig = (sae_recon_tokens * self.act_std + self.act_mean)
                
                # Calculate losses
                vit_mse = F.mse_loss(vit_recon, vol)
                sae_mse = F.mse_loss(data['sae_reconstruction'], data['vit_tokens_wh'])
                
                results.append({
                    'volume_idx': i,
                    'original': vol.cpu(),
                    'vit_reconstruction': vit_recon.cpu(),
                    'sae_tokens_original': data['vit_tokens'],
                    'sae_tokens_reconstructed': sae_recon_tokens_orig.cpu(),
                    'vit_mse': vit_mse.item(),
                    'sae_mse': sae_mse.item(),
                    'mask': mask.cpu() if mask is not None else None
                })
        
        return results
    
    def create_interactive_visualization(self, reconstruction_results, output_path="sae_analysis_viewer.html"):
        """Create interactive HTML visualization with sliders."""
        print("Creating interactive visualization...")
        
        # Create HTML with embedded JavaScript
        html_content = self._generate_html_viewer(reconstruction_results)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Interactive visualization saved to {output_path}")
        return output_path
    
    def _generate_html_viewer(self, reconstruction_results):
        """Generate HTML content for interactive viewer."""
        # Convert numpy arrays to nested lists for JSON serialization
        data_for_js = []
        for result in reconstruction_results:
            # Convert 3D volumes to 2D slices
            vol_orig = result['original'][0, 0].numpy()  # Remove batch and channel dims
            vol_vit = result['vit_reconstruction'][0, 0].numpy()
            
            # Calculate difference
            vol_diff = np.abs(vol_orig - vol_vit)
            
            data_for_js.append({
                'volume_idx': result['volume_idx'],
                'original': vol_orig.tolist(),
                'vit_reconstruction': vol_vit.tolist(),
                'difference': vol_diff.tolist(),
                'vit_mse': result['vit_mse'],
                'sae_mse': result['sae_mse'],
                'mask': result['mask'][0, 0].numpy().tolist() if result['mask'] is not None else None
            })
        
        return f'''
<!DOCTYPE html>
<html>
<head>
    <title>SAE Analysis Viewer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .controls {{ margin: 20px 0; }}
        .slider-container {{ margin: 10px 0; }}
        .metrics {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .plot-container {{ display: inline-block; width: 300px; margin: 10px; }}
    </style>
</head>
<body>
    <h1>SAE Analysis: Reconstruction Quality Viewer</h1>
    
    <div class="controls">
        <div class="slider-container">
            <label>Volume: </label>
            <input type="range" id="volumeSlider" min="0" max="{len(data_for_js)-1}" value="0" step="1">
            <span id="volumeDisplay">0</span>
        </div>
        
        <div class="slider-container">
            <label>Slice: </label>
            <input type="range" id="sliceSlider" min="0" max="95" value="48" step="1">
            <span id="sliceDisplay">48</span>
        </div>
        
        <div class="metrics">
            <strong>Quality Metrics:</strong><br>
            VIT MSE: <span id="vitMse">-</span><br>
            SAE MSE: <span id="saeMse">-</span>
        </div>
    </div>
    
    <div id="plotContainer">
        <div class="plot-container">
            <div id="originalPlot"></div>
            <h3>Original</h3>
        </div>
        
        <div class="plot-container">
            <div id="vitPlot"></div>
            <h3>VIT Reconstruction</h3>
        </div>
        
        <div class="plot-container">
            <div id="diffPlot"></div>
            <h3>Difference</h3>
        </div>
        
        <div class="plot-container">
            <div id="maskPlot"></div>
            <h3>Segmentation Mask</h3>
        </div>
    </div>
    
    <script>
        const data = {repr(data_for_js)};
        
        let currentVolume = 0;
        let currentSlice = 48;
        
        function updatePlots() {{
            const volumeData = data[currentVolume];
            const slice = currentSlice;
            
            // Update metrics
            document.getElementById('vitMse').textContent = volumeData.vit_mse.toFixed(6);
            document.getElementById('saeMse').textContent = volumeData.sae_mse.toFixed(6);
            
            // Create plots
            const plotConfig = {{
                displayModeBar: false,
                staticPlot: false
            }};
            
            const layout = {{
                width: 280,
                height: 280,
                margin: {{l: 40, r: 40, t: 40, b: 40}},
                xaxis: {{showticklabels: false}},
                yaxis: {{showticklabels: false}}
            }};
            
            // Original
            Plotly.newPlot('originalPlot', [{{
                z: volumeData.original[slice],
                type: 'heatmap',
                colorscale: 'Greys',
                zmin: 0,
                zmax: 1,
                showscale: false,
                autocolorscale: false
            }}], layout, plotConfig);
            
            // VIT Reconstruction
            Plotly.newPlot('vitPlot', [{{
                z: volumeData.vit_reconstruction[slice],
                type: 'heatmap',
                colorscale: 'Greys',
                zmin: 0,
                zmax: 1,
                showscale: false,
                autocolorscale: false
            }}], layout, plotConfig);
            
            // Difference
            Plotly.newPlot('diffPlot', [{{
                z: volumeData.difference[slice],
                type: 'heatmap',
                colorscale: 'Hot',
                zmin: 0,
                zmax: 1,
                showscale: false,
                autocolorscale: false
            }}], layout, plotConfig);
            
            // Mask
            if (volumeData.mask) {{
                Plotly.newPlot('maskPlot', [{{
                    z: volumeData.mask[slice],
                    type: 'heatmap',
                    colorscale: 'Viridis',
                    zmin: 0,
                    zmax: 3,
                    showscale: false,
                    autocolorscale: false
                }}], layout, plotConfig);
            }}
        }}
        
        // Event listeners
        document.getElementById('volumeSlider').addEventListener('input', function() {{
            currentVolume = parseInt(this.value);
            document.getElementById('volumeDisplay').textContent = currentVolume;
            updatePlots();
        }});
        
        document.getElementById('sliceSlider').addEventListener('input', function() {{
            currentSlice = parseInt(this.value);
            document.getElementById('sliceDisplay').textContent = currentSlice;
            updatePlots();
        }});
        
        // Initial plot
        updatePlots();
    </script>
</body>
</html>
        '''
    
    def analyze_feature_geometry(self, selectivity_results, num_components=10):
        """Analyze the geometry of learned features using PCA and clustering."""
        print("Analyzing feature geometry...")
        
        # Get SAE weights for active neurons
        active_neurons = list(selectivity_results.keys())
        if len(active_neurons) == 0:
            print("No active neurons found!")
            return None
        
        # Extract encoder weights
        encoder_weights = self.sae.encoder_weight[active_neurons].detach().cpu().numpy()
        
        # PCA analysis
        pca = PCA(n_components=min(num_components, encoder_weights.shape[0]))
        pca_features = pca.fit_transform(encoder_weights)
        
        # Clustering analysis
        n_clusters = min(8, len(active_neurons))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(encoder_weights)
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PCA Projection', 'Feature Similarity', 
                          'Cluster Analysis', 'Explained Variance'),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # PCA projection
        fig.add_trace(
            go.Scatter(
                x=pca_features[:, 0],
                y=pca_features[:, 1],
                mode='markers+text',
                text=[f"N{n}" for n in active_neurons],
                textposition="top center",
                marker=dict(size=8, color=clusters, colorscale='Viridis'),
                name='Neurons'
            ),
            row=1, col=1
        )
        
        # Feature similarity heatmap
        similarity_matrix = np.corrcoef(encoder_weights)
        fig.add_trace(
            go.Heatmap(
                z=similarity_matrix,
                colorscale='RdBu',
                name='Similarity'
            ),
            row=1, col=2
        )
        
        # Cluster analysis
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_features = pca_features[cluster_mask]
            fig.add_trace(
                go.Scatter(
                    x=cluster_features[:, 0],
                    y=cluster_features[:, 1],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    marker=dict(size=10)
                ),
                row=2, col=1
            )
        
        # Explained variance
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                y=pca.explained_variance_ratio_,
                name='Explained Variance'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="SAE Feature Geometry Analysis",
            height=800,
            showlegend=True
        )
        
        return fig, {
            'pca': pca,
            'clusters': clusters,
            'similarity_matrix': similarity_matrix,
            'explained_variance': pca.explained_variance_ratio_
        }
    
    def generate_feature_dashboard(self, selectivity_results, output_path="sae_dashboard.html"):
        """Generate comprehensive dashboard with all analysis results."""
        print("Generating feature dashboard...")
        
        # Create multiple visualizations
        visualizations = []
        
        # 1. Selectivity distribution
        selectivity_scores = [r['selectivity_index'] for r in selectivity_results.values() if r is not None]
        dominance_scores = [r['dominance_score'] for r in selectivity_results.values() if r is not None]
        
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(x=selectivity_scores, name='Selectivity Index', nbinsx=20))
        fig1.update_layout(title="Distribution of Selectivity Scores", xaxis_title="Selectivity Index")
        visualizations.append(fig1.to_html(include_plotlyjs='cdn', div_id="selectivity_dist"))
        
        # 2. Monosemantic neurons summary
        mono_summary = {}
        for obj_type, neurons in self.monosemantic_neurons.items():
            mono_summary[obj_type] = len(neurons)
        
        fig2 = go.Figure(data=[go.Bar(x=list(mono_summary.keys()), y=list(mono_summary.values()))])
        fig2.update_layout(title="Monosemantic Neurons by Object Type", 
                          xaxis_title="Object Type", yaxis_title="Number of Neurons")
        visualizations.append(fig2.to_html(include_plotlyjs='cdn', div_id="mono_summary"))
        
        # 3. Feature geometry analysis
        geom_fig, geom_data = self.analyze_feature_geometry(selectivity_results)
        visualizations.append(geom_fig.to_html(include_plotlyjs='cdn', div_id="feature_geometry"))
        
        # Combine all visualizations
        dashboard_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAE Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 30px 0; }}
                .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>SAE Monosemanticity Analysis Dashboard</h1>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="summary">
                    <p><strong>Total Neurons Analyzed:</strong> {len(selectivity_results)}</p>
                    <p><strong>Active Neurons:</strong> {len([r for r in selectivity_results.values() if r is not None])}</p>
                    <p><strong>Monosemantic Neurons Found:</strong> {sum(len(neurons) for neurons in self.monosemantic_neurons.values())}</p>
                    <p><strong>Mean Selectivity:</strong> {np.mean(selectivity_scores):.3f}</p>
                    <p><strong>Mean Dominance:</strong> {np.mean(dominance_scores):.3f}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Selectivity Distribution</h2>
                {visualizations[0]}
            </div>
            
            <div class="section">
                <h2>Monosemantic Neurons by Object Type</h2>
                {visualizations[1]}
            </div>
            
            <div class="section">
                <h2>Feature Geometry Analysis</h2>
                {visualizations[2]}
            </div>
            
            <div class="section">
                <h2>Detailed Monosemantic Neurons</h2>
                {''.join(self._generate_neuron_details_html())}
            </div>
        </body>
        </html>
        '''
        
        with open(output_path, 'w') as f:
            f.write(dashboard_html)
        
        print(f"Dashboard saved to {output_path}")
        return output_path
    
    def _generate_neuron_details_html(self):
        """Generate HTML for detailed neuron information."""
        html_parts = []
        
        for obj_type, neurons in self.monosemantic_neurons.items():
            html_parts.append(f'<h3>{obj_type.title()} Detectors</h3>')
            html_parts.append('<table border="1" style="border-collapse: collapse; width: 100%;">')
            html_parts.append('<tr><th>Neuron ID</th><th>Selectivity</th><th>Dominance</th><th>Active Patches</th><th>Sparsity</th></tr>')
            
            for neuron in neurons[:10]:  # Show top 10
                metrics = neuron['metrics']
                html_parts.append(f'''
                <tr>
                    <td>{neuron['neuron_idx']}</td>
                    <td>{neuron['selectivity']:.3f}</td>
                    <td>{neuron['dominance']:.3f}</td>
                    <td>{metrics['total_active_patches']}</td>
                    <td>{metrics['sparsity']:.3f}</td>
                </tr>
                ''')
            
            html_parts.append('</table>')
        
        return html_parts


def load_models(args):
    """Load VIT and SAE models from checkpoints."""
    device = get_device()
    
    # Load VIT model
    print("Loading VIT model...")
    archs = {
        "small": mae_vit_3d_small_conv,
        "base": mae_vit_3d_base_conv,
        "large": mae_vit_3d_large_conv,
        "hemibrain_optimal": mae_vit_3d_hemibrain_optimal_conv,
        "base_patch_conv": mae_vit_3d_base_patch_conv,
    }
    
    vit = archs[args.model_arch](
        volume_size=(args.img_size,) * 3,
        patch_size=args.patch_size,
        norm_pix_loss=False,
        mask_ratio=0.0
    ).to(device)
    
    # Load VIT checkpoint
    vit_ckpt = torch.load(args.vit_checkpoint, map_location="cpu")
    vit_state_dict = vit_ckpt.get("model_state_dict") or vit_ckpt.get("ema_state_dict")
    
    # Clean state dict
    clean_state_dict = {}
    for key, value in vit_state_dict.items():
        if key.startswith('_orig_mod.'):
            clean_key = key[len('_orig_mod.'):]
            clean_state_dict[clean_key] = value
        else:
            clean_state_dict[key] = value
    
    vit.load_state_dict(clean_state_dict, strict=False)
    vit.eval()
    
    # Compile VIT for speed
    if args.compile_models:
        vit = torch.compile(vit, backend="inductor", mode="default")
    
    # Load SAE model
    print("Loading SAE model...")
    sae_ckpt = torch.load(args.sae_checkpoint, map_location="cpu")
    
    input_dim = sae_ckpt["input_dim"]
    latent_dim = sae_ckpt["latent_dim"]
    
    if args.sae_variant == "linear":
        sae = LinearSAE(input_dim, latent_dim, activation=args.activation)
    elif args.sae_variant == "gated":
        sae = GatedSAE(input_dim, latent_dim)
    else:
        raise ValueError(f"Unknown SAE variant: {args.sae_variant}")
    
    sae.encoder_weight.data = sae_ckpt["sae_weight"]
    sae.encoder_bias.data = sae_ckpt["encoder_bias"]
    sae.decoder_weight.data = sae_ckpt["decoder_weight"]
    sae.decoder_bias.data = sae_ckpt["decoder_bias"]
    sae.to(device)
    sae.eval()
    
    # Compile SAE for speed
    if args.compile_models:
        sae = torch.compile(sae, backend="inductor", mode="default")
    
    # Create token extractor
    token_extractor = TokenExtractor(vit, args.layer, extract_from=args.extract_from)
    
    # Compute whitening statistics
    print("Computing whitening statistics...")
    dummy_volume = torch.randn(1, 1, args.img_size, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        dummy_tokens = token_extractor.extract_tokens(dummy_volume)
        act_mean = torch.zeros(1, dummy_tokens.shape[1]).to(device)
        act_std = torch.ones(1, dummy_tokens.shape[1]).to(device)
        
        # Use a few batches to estimate statistics
        if hasattr(sae_ckpt, 'act_mean') and hasattr(sae_ckpt, 'act_std'):
            act_mean = sae_ckpt['act_mean'].to(device)
            act_std = sae_ckpt['act_std'].to(device)
    
    return vit, sae, token_extractor, act_mean, act_std, device


def load_data(args):
    """Load volume and mask data from tar shards."""
    print("Loading data...")
    
    # Load volume shards
    volume_shards = sorted(Path(args.shard_dir).glob("shard_*.tar"))
    mask_shards = sorted(Path(args.shard_dir).glob("masks/shard_*.tar"))
    
    print(f"Found {len(volume_shards)} volume shards, {len(mask_shards)} mask shards")
    
    # Use a subset for analysis
    analysis_shards = volume_shards[:args.max_shards]
    analysis_mask_shards = mask_shards[:args.max_shards]
    
    volumes = []
    masks = []
    
    for vol_shard, mask_shard in zip(analysis_shards, analysis_mask_shards):
        # Load volumes
        vol_count = 0
        with tarfile.open(vol_shard, "r") as tar:
            for member in tar:
                if vol_count >= args.max_volumes_per_shard:
                    break
                    
                if member.isfile():
                    vol_data = tar.extractfile(member).read()
                    vol = np.frombuffer(vol_data, dtype=np.float32)
                    vol = vol.reshape(args.img_size, args.img_size, args.img_size)
                    # Add both channel and batch dimensions: (D, H, W) -> (1, 1, D, H, W)
                    volumes.append(torch.from_numpy(vol.copy()).unsqueeze(0).unsqueeze(0))
                    vol_count += 1
        
        # Load masks
        mask_count = 0
        with tarfile.open(mask_shard, "r") as tar:
            for member in tar:
                if mask_count >= args.max_volumes_per_shard:
                    break
                    
                if member.isfile():
                    mask_data = tar.extractfile(member).read()
                    mask = np.frombuffer(mask_data, dtype=np.uint8)
                    mask = mask.reshape(args.img_size, args.img_size, args.img_size)
                    # Add both channel and batch dimensions: (D, H, W) -> (1, 1, D, H, W)
                    masks.append(torch.from_numpy(mask.copy()).unsqueeze(0).unsqueeze(0))
                    mask_count += 1
    
    print(f"Loaded {len(volumes)} volumes and {len(masks)} masks")
    return volumes, masks


def main():
    parser = argparse.ArgumentParser(description="Comprehensive SAE Analysis")
    
    # Model paths
    parser.add_argument("--vit_checkpoint", required=True, help="Path to VIT checkpoint")
    parser.add_argument("--sae_checkpoint", required=True, help="Path to SAE checkpoint")
    parser.add_argument("--shard_dir", required=True, help="Directory with volume shards")
    
    # Model config
    parser.add_argument("--model_arch", default="base", choices=["small", "base", "large", "hemibrain_optimal", "base_patch_conv"])
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--extract_from", choices=["patchembed", "encoder"], default="encoder")
    parser.add_argument("--sae_variant", choices=["linear", "gated"], default="linear")
    parser.add_argument("--activation", choices=["relu", "gelu"], default="relu")
    
    # Analysis parameters
    parser.add_argument("--max_shards", type=int, default=2, help="Maximum number of shards to analyze")
    parser.add_argument("--max_volumes_per_shard", type=int, default=100, help="Maximum volumes per shard")
    parser.add_argument("--min_selectivity", type=float, default=0.7, help="Minimum selectivity for monosemantic neurons")
    parser.add_argument("--min_dominance", type=float, default=0.6, help="Minimum dominance for monosemantic neurons")
    parser.add_argument("--num_reconstruction_samples", type=int, default=20, help="Number of samples for reconstruction analysis")
    
    # Output
    parser.add_argument("--output_dir", default="sae_analysis_results", help="Output directory")
    parser.add_argument("--compile_models", action="store_true", help="Compile models with torch.compile")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="bf16")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load models
    vit, sae, token_extractor, act_mean, act_std, device = load_models(args)
    
    # Load data
    volumes, masks = load_data(args)
    
    # Create analyzer
    analyzer = SAEAnalyzer(vit, sae, token_extractor, act_mean, act_std, device)
    
    # Run analysis
    print("\n" + "="*60)
    print("STARTING SAE MONOSEMANTICITY ANALYSIS")
    print("="*60)
    
    # 1. Analyze neuron selectivity
    print("\n1. Analyzing neuron selectivity...")
    selectivity_results = analyzer.analyze_neuron_selectivity(volumes, masks, max_batches=args.max_shards)
    
    # 2. Find monosemantic neurons
    print("\n2. Finding monosemantic neurons...")
    monosemantic_neurons = analyzer.find_monosemantic_neurons(
        selectivity_results, 
        min_selectivity=args.min_selectivity,
        min_dominance=args.min_dominance
    )
    
    # 3. Analyze reconstruction quality
    print("\n3. Analyzing reconstruction quality...")
    reconstruction_results = analyzer.analyze_reconstruction_quality(
        volumes, masks, 
        num_samples=args.num_reconstruction_samples
    )
    
    # 4. Create visualizations
    print("\n4. Creating visualizations...")
    
    # Interactive viewer
    viewer_path = Path(args.output_dir) / "reconstruction_viewer.html"
    analyzer.create_interactive_visualization(reconstruction_results, viewer_path)
    
    # Comprehensive dashboard
    dashboard_path = Path(args.output_dir) / "sae_dashboard.html"
    analyzer.generate_feature_dashboard(selectivity_results, dashboard_path)
    
    # Save results
    print("\n5. Saving results...")
    results_path = Path(args.output_dir) / "analysis_results.pt"
    torch.save({
        'selectivity_results': selectivity_results,
        'monosemantic_neurons': monosemantic_neurons,
        'reconstruction_results': reconstruction_results,
        'config': vars(args)
    }, results_path)
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    print(f"Interactive viewer: {viewer_path}")
    print(f"Dashboard: {dashboard_path}")
    print(f"Raw results: {results_path}")
    
    total_neurons = len(selectivity_results)
    active_neurons = len([r for r in selectivity_results.values() if r is not None])
    mono_neurons = sum(len(neurons) for neurons in monosemantic_neurons.values())
    
    print(f"\nSummary:")
    print(f"  Total neurons analyzed: {total_neurons}")
    print(f"  Active neurons: {active_neurons}")
    print(f"  Monosemantic neurons: {mono_neurons}")
    print(f"  Monosemanticity rate: {mono_neurons/active_neurons*100:.1f}%")
    
    # Cleanup
    token_extractor.cleanup()


if __name__ == "__main__":
    main() 