#!/usr/bin/env python3
"""
Create interactive HTML viewer for model reconstructions from a checkpoint.
"""

import argparse
import json
import base64
import glob
import os
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, IterableDataset
import tarfile

from vit_3d import (
    mae_vit_3d_small, mae_vit_3d_base, mae_vit_3d_large, mae_vit_3d_huge,
    mae_vit_3d_hemibrain_optimal,
    mae_vit_3d_small_conv, mae_vit_3d_base_conv, mae_vit_3d_large_conv,
    mae_vit_3d_hemibrain_optimal_conv,
    get_device,
)

# Copied from vol_train.py for data loading
class TarShardDataset(IterableDataset):
    """Tar dataset with I/O optimizations."""
    def __init__(self, shards, volume_size: int, shuffle: bool = False, 
                 vols_per_shard: int = 16384):
        self.shards = shards
        self.shuffle = shuffle
        self.vols_per_shard = vols_per_shard
        self.volume_size = volume_size
        
        print(f"Dataset: {len(shards)} shards, {self.vols_per_shard} vols/shard, shuffle={shuffle}")

    def __len__(self):
        return len(self.shards) * self.vols_per_shard

    def _iter_shard(self, path):
        try:
            with tarfile.open(path, "r|", bufsize=32*1024*1024) as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    buf = tar.extractfile(member).read()
                    vol = np.frombuffer(buf, dtype=np.float32)
                    vol = vol.reshape(self.volume_size, self.volume_size, self.volume_size)
                    volume_tensor = torch.from_numpy(vol).contiguous().unsqueeze(0)
                    yield volume_tensor
        except Exception as e:
            print(f"Error reading shard {path}: {e}")
            return

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        shard_list = self.shards.copy()
        if self.shuffle:
            random.shuffle(shard_list)
        
        if worker:
            worker_shards = shard_list[worker.id::worker.num_workers]
        else:
            worker_shards = shard_list
            
        for shard_path in worker_shards:
            yield from self._iter_shard(shard_path)


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the most recently modified checkpoint file."""
    list_of_files = glob.glob(f'{checkpoint_dir}/*.pt')
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def load_model_from_checkpoint(checkpoint_path, device):
    """Load a model from a training checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    archs = {
        "small": mae_vit_3d_small, "base": mae_vit_3d_base,
        "large": mae_vit_3d_large, "huge": mae_vit_3d_huge,
        "hemibrain_optimal": mae_vit_3d_hemibrain_optimal,
        "small_conv": mae_vit_3d_small_conv, "base_conv": mae_vit_3d_base_conv,
        "large_conv": mae_vit_3d_large_conv,
        "hemibrain_optimal_conv": mae_vit_3d_hemibrain_optimal_conv
    }
    
    model = archs[config['model_arch']](
        volume_size=(config['img_size'],)*3,
        patch_size=config['patch_size'],
        norm_pix_loss=config['norm_pix_loss'],
    ).to(device)

    # The model was trained with torch.compile, which can prefix state_dict keys.
    # We remove the prefix to ensure compatibility.
    from collections import OrderedDict
    def strip_prefix(state_dict):
        if state_dict is None:
            return None
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                name = k[len('_orig_mod.'):]  # remove `_orig_mod.`
            elif k.startswith('module.'):
                name = k[len('module.'):] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        return new_state_dict

    # Use EMA state if available, otherwise regular model state
    state_to_load = checkpoint.get('ema_state_dict') or checkpoint.get('model_state_dict')
    
    if state_to_load:
        if checkpoint.get('ema_state_dict'):
            print("Loading EMA model state.")
        else:
            print("Loading standard model state.")
        
        cleaned_state_dict = strip_prefix(state_to_load)
        model.load_state_dict(cleaned_state_dict)
    else:
        raise KeyError("Checkpoint does not contain 'ema_state_dict' or 'model_state_dict'")
        
    model.eval()
    print("Model loaded successfully.")
    return model, config

@torch.inference_mode()
def generate_reconstructions(model, loader, device, num_examples=8, mask_ratio=0.75):
    """Generate reconstructions and other visualizations for a number of examples."""
    print(f"Generating reconstructions for {num_examples} examples...")
    reconstructions = []
    
    for i, batch in enumerate(loader):
        if i >= num_examples:
            break
        
        print(f"  Processing example {i+1}/{num_examples}")
        batch = batch.to(device)
        
        # Get model output
        loss, y, mask, _ = model(batch, mask_ratio=mask_ratio)
        
        # Reconstruct the full image from patches
        y = model.unpatchify(y)
        
        # Create mask for visualization (1 for removed, 0 for visible)
        mask = mask.detach()
        patch_vol = model.encoder.patch_size[0] * model.encoder.patch_size[1] * model.encoder.patch_size[2]
        mask_for_vis = mask.unsqueeze(-1).repeat(1, 1, patch_vol)
        mask_for_vis = model.unpatchify(mask_for_vis)
        
        original_vol = batch.squeeze().cpu().numpy()
        reconstructed_vol = y.squeeze().cpu().numpy()
        
        reconstructions.append({
            'original': original_vol,
            'reconstructed': reconstructed_vol,
            'diff': np.abs(original_vol - reconstructed_vol),
            'mask_vis': mask_for_vis.squeeze().cpu().numpy(),
            'loss': loss.item()
        })
        
    return reconstructions

def plot_slice_to_base64(slice_data, title, cmap='gray', vmin=None, vmax=None):
    """Generic function to plot a 2D slice and return a base64 string."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(slice_data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    ax.set_title(title, fontsize=10, pad=5)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=90)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def create_vol_images(volume, title_prefix, cmap='gray'):
    """Convert a 3D volume to a list of base64 encoded images."""
    images = []
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)
    vmin, vmax = (0, 1) if cmap == 'gray' else (None, None)
    
    for z in range(volume.shape[0]):
        title = f"{title_prefix} (Slice {z})"
        images.append(plot_slice_to_base64(vol_norm[z], title, cmap=cmap, vmin=vmin, vmax=vmax))
    return images

def create_colored_mask_images(original_vol, mask_vol):
    """Create red/green overlay showing kept vs. removed patches."""
    images = []
    vol_norm = (original_vol - original_vol.min()) / (original_vol.max() - original_vol.min() + 1e-6)
    
    for z in range(original_vol.shape[0]):
        original_slice = vol_norm[z]
        mask_slice = mask_vol[z]  # 1 for removed, 0 for kept
        
        rgb_slice = np.zeros((*original_slice.shape, 3), dtype=np.float32)
        
        # Green channel for visible patches (mask == 0)
        rgb_slice[..., 1] = np.where(mask_slice == 0, original_slice, 0)
        
        # Red channel for removed patches (mask == 1)
        rgb_slice[..., 0] = np.where(mask_slice == 1, original_slice, 0)
        
        title = f"Masked Input (Slice {z})"
        images.append(plot_slice_to_base64(rgb_slice, title))
    return images

def create_interactive_html(reconstructions, output_file="reconstruction_viewer.html"):
    """Create an interactive HTML viewer for reconstructions."""
    print(f"Creating interactive HTML viewer: {output_file}")
    
    all_data = []
    for i, recon_data in enumerate(reconstructions):
        print(f"Processing volume {i+1}/{len(reconstructions)} for HTML...")
        
        original_imgs = create_vol_images(recon_data['original'], "Original", cmap='gray')
        recon_imgs = create_vol_images(recon_data['reconstructed'], "Reconstruction", cmap='gray')
        diff_imgs = create_vol_images(recon_data['diff'], "Difference", cmap='magma')
        colored_mask_imgs = create_colored_mask_images(recon_data['original'], recon_data['mask_vis'])
        
        all_data.append({
            'original_images': original_imgs,
            'reconstructed_images': recon_imgs,
            'diff_images': diff_imgs,
            'colored_mask_images': colored_mask_imgs,
            'stats': {
                'loss': recon_data['loss'],
                'original_range': [float(recon_data['original'].min()), float(recon_data['original'].max())],
            }
        })
        
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Volume Reconstruction Viewer</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background-color: #f0f2f5; color: #1c1e21; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 25px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 25px; border-bottom: 1px solid #ddd; padding-bottom: 15px; }
        .header h1 { font-size: 2em; }
        .controls { display: flex; justify-content: center; align-items: center; gap: 30px; margin-bottom: 25px; padding: 20px; background-color: #f8f9fa; border-radius: 8px; }
        .control-group { display: flex; align-items: center; gap: 10px; font-size: 1.1em; }
        .slider { width: 400px; }
        select, input[type=range] { transform: scale(1.1); }
        .images-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 25px; }
        .image-container { text-align: center; background-color: #fff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
        .image-container img { max-width: 100%; height: auto; border-radius: 5px; }
        .image-container p { margin-top: 8px; font-size: 1em; font-weight: 600; color: #333; }
        .stats { background-color: #e9ecef; padding: 20px; border-radius: 8px; margin-top: 20px; text-align: center; font-size: 1.1em; }
        .stats div { margin-bottom: 5px; }
        .legend { text-align: center; margin-bottom: 20px; padding: 10px; background-color: #fffbe6; border: 1px solid #ffe58f; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>3D Volume Reconstruction Viewer</h1>
            <p>Interactive side-by-side comparison of MAE model reconstructions</p>
        </div>
        
        <div class="legend">
            <strong>Masked Input Legend:</strong> 
            <span style="color: #28a745; font-weight: bold;">Green = Visible Patches</span>, 
            <span style="color: #dc3545; font-weight: bold;">Red = Masked (Removed) Patches</span>
        </div>

        <div class="controls">
            <div class="control-group">
                <label for="volumeSelect">Example:</label>
                <select id="volumeSelect" onchange="changeVolume()"></select>
            </div>
            <div class="control-group">
                <label for="sliceSlider">Slice (Z):</label>
                <input type="range" id="sliceSlider" class="slider" min="0" max="95" value="48" oninput="updateSlice(this.value)">
                <span id="sliceValue" style="font-weight: bold; min-width: 25px; text-align: center;">48</span>
            </div>
        </div>
        
        <div class="images-grid">
            <div class="image-container">
                <img id="originalImage" src="" alt="Original slice">
                <p>Original</p>
            </div>
            <div class="image-container">
                <img id="reconstructedImage" src="" alt="Reconstructed slice">
                <p>Reconstruction</p>
            </div>
            <div class="image-container">
                <img id="maskImage" src="" alt="Masked Input slice">
                <p>Masked Input</p>
            </div>
            <div class="image-container">
                <img id="diffImage" src="" alt="Difference slice">
                <p>Difference (Original - Recon)</p>
            </div>
        </div>
        
        <div class="stats">
            <h3>Example Statistics</h3>
            <div><strong>Reconstruction Loss (MSE):</strong> <span id="statLoss">-</span></div>
            <div><strong>Original Value Range:</strong> <span id="statRange">-</span></div>
        </div>
    </div>

    <script>
        const reconData = """ + json.dumps(all_data) + """;
        
        let currentVolume = 0;
        let currentSlice = 48;
        
        function initializeViewer() {
            const volumeSelect = document.getElementById('volumeSelect');
            reconData.forEach((vol, i) => {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `Example ${i + 1}`;
                volumeSelect.appendChild(option);
            });
            const sliceSlider = document.getElementById('sliceSlider');
            sliceSlider.max = reconData[0].original_images.length - 1;
            sliceSlider.value = Math.floor(sliceSlider.max / 2);
            currentSlice = parseInt(sliceSlider.value);

            updateDisplay();
        }
        
        function changeVolume() {
            currentVolume = parseInt(document.getElementById('volumeSelect').value);
            updateDisplay();
        }
        
        function updateSlice(slice) {
            currentSlice = parseInt(slice);
            updateDisplay();
        }
        
        function updateDisplay() {
            const vol = reconData[currentVolume];
            
            document.getElementById('originalImage').src = 'data:image/png;base64,' + vol.original_images[currentSlice];
            document.getElementById('reconstructedImage').src = 'data:image/png;base64,' + vol.reconstructed_images[currentSlice];
            document.getElementById('diffImage').src = 'data:image/png;base64,' + vol.diff_images[currentSlice];
            document.getElementById('maskImage').src = 'data:image/png;base64,' + vol.colored_mask_images[currentSlice];

            document.getElementById('sliceValue').textContent = currentSlice;
            
            const stats = vol.stats;
            document.getElementById('statLoss').textContent = stats.loss.toFixed(6);
            document.getElementById('statRange').textContent = 
                `[${stats.original_range[0].toFixed(3)}, ${stats.original_range[1].toFixed(3)}]`;
        }
        
        document.addEventListener('keydown', function(event) {
            const sliceSlider = document.getElementById('sliceSlider');
            const maxSlice = parseInt(sliceSlider.max);
            if (event.key === 'ArrowUp' && currentSlice < maxSlice) {
                currentSlice++;
                sliceSlider.value = currentSlice;
                updateSlice(currentSlice);
            } else if (event.key === 'ArrowDown' && currentSlice > 0) {
                currentSlice--;
                sliceSlider.value = currentSlice;
                updateSlice(currentSlice);
            }
        });
        
        window.onload = initializeViewer;
    </script>
</body>
</html>
"""
    with open(output_file, 'w') as f:
        f.write(html_template)
    
    print(f"✓ Interactive viewer saved as {output_file}")
    print(f"  File size: {len(html_template) / 1024 / 1024:.1f} MB")

def main():
    P = argparse.ArgumentParser(description="Generate interactive viewers for model reconstructions.")
    P.add_argument("--checkpoint_path", type=str, default=None,
                   help="Path to model checkpoint. If None, finds the latest in `checkpoints/`.")
    P.add_argument("--shard_dir", required=True, help="Directory with .tar shards for validation.")
    P.add_argument("--num_examples", type=int, default=8, help="Number of examples to visualize.")
    P.add_argument("--mask_ratio", type=float, default=0.75, help="Masking ratio for the model.")
    P.add_argument("--output_file", type=str, default="reconstruction_viewer.html",
                   help="Name of the output HTML file.")
    P.add_argument("--val_split", type=float, default=0.02, help="Fraction of shards for validation.")
    
    args = P.parse_args()
    
    device = get_device()
    
    checkpoint_path = args.checkpoint_path or find_latest_checkpoint()
    if not checkpoint_path:
        raise FileNotFoundError("No checkpoint found. Please specify --checkpoint_path or place one in checkpoints/")
        
    model, config = load_model_from_checkpoint(checkpoint_path, device)
    
    # Setup dataloader with validation shards
    shards = sorted(Path(args.shard_dir).expanduser().glob("shard*.tar"))
    n_val  = max(1, int(len(shards) * args.val_split))
    val_shards = shards[:n_val]
    
    if not val_shards:
        raise FileNotFoundError(f"No validation shards found in {args.shard_dir} with val_split={args.val_split}")
        
    dataset = TarShardDataset(
        val_shards, config['img_size'], shuffle=True, vols_per_shard=config.get('vols_per_shard', 16384)
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    # Generate data
    reconstructions = generate_reconstructions(model, loader, device, args.num_examples, args.mask_ratio)
    
    # Create HTML
    create_interactive_html(reconstructions, args.output_file)
    
    print(f"\n✓ Complete! Open {args.output_file} in your browser.")

if __name__ == "__main__":
    main() 