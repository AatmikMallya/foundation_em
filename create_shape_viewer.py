#!/usr/bin/env python3
"""
Create interactive HTML viewer for volumes with membranes, spheres, and cubes
"""

import json
import base64
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from vol_generator import MembraneGen

def generate_combination_examples():
    """Generate examples of each structure combination"""
    print("Generating examples of each combination...")
    gen = MembraneGen(generate_masks=True, equal_combinations=False)  # We'll control combinations manually
    
    # Define all 8 combinations with descriptive names
    combinations = [
        {'name': 'Background Only', 'membrane': False, 'spheres': False, 'cubes': False},
        {'name': 'Membranes Only', 'membrane': True, 'spheres': False, 'cubes': False},
        {'name': 'Spheres Only', 'membrane': False, 'spheres': True, 'cubes': False},
        {'name': 'Cubes Only', 'membrane': False, 'spheres': False, 'cubes': True},
        {'name': 'Membranes + Spheres', 'membrane': True, 'spheres': True, 'cubes': False},
        {'name': 'Membranes + Cubes', 'membrane': True, 'spheres': False, 'cubes': True},
        {'name': 'Spheres + Cubes', 'membrane': False, 'spheres': True, 'cubes': True},
        {'name': 'All Structures', 'membrane': True, 'spheres': True, 'cubes': True},
    ]
    
    volumes = []
    masks = []
    combo_info = []
    
    # Generate one example of each combination
    for i, combo in enumerate(combinations):
        print(f"  Generating {combo['name']} (seed {i*100 + 42})")
        
        # Temporarily modify the generator to force this combination
        old_combinations = gen.combinations
        gen.combinations = [combo]  # Force this specific combination
        
        seed = i * 100 + 42
        vol_bytes, mask_bytes = gen(seed)
        
        volume = np.frombuffer(vol_bytes, dtype=np.float32).reshape(96, 96, 96)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
        
        volumes.append(volume)
        masks.append(mask)
        combo_info.append(combo)
        
        # Print stats
        unique_labels = np.unique(mask)
        counts = np.bincount(mask.flatten())
        print(f"    Labels: {unique_labels}, Counts: {counts[:len(unique_labels)]}")
        
        # Restore original combinations
        gen.combinations = old_combinations
    
    return volumes, masks, combo_info

def create_static_examples():
    """Create static slice examples showing different combinations"""
    print("Creating static slice examples...")
    
    volumes, masks, combo_info = generate_combination_examples()
    
    # Create a large figure showing middle slices of each combination
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Create colormap for masks
    colors = ['black', 'blue', 'red', 'orange']  # bg, membrane, sphere, cube
    cmap = mcolors.ListedColormap(colors)
    
    for i, (volume, mask, combo) in enumerate(zip(volumes, masks, combo_info)):
        ax = axes[i]
        
        # Show middle slice (z=48)
        z_slice = 48
        vol_norm = (volume - volume.min()) / (volume.max() - volume.min())
        
        # Show overlay
        ax.imshow(vol_norm[z_slice], cmap='gray', vmin=0, vmax=1, alpha=0.8)
        
        # Overlay shapes with transparency
        membrane_mask = mask[z_slice] == 1
        sphere_mask = mask[z_slice] == 2
        cube_mask = mask[z_slice] == 3
        
        if membrane_mask.any():
            ax.imshow(np.where(membrane_mask, 1, np.nan), cmap='Blues', alpha=0.6, vmin=0, vmax=1)
        if sphere_mask.any():
            ax.imshow(np.where(sphere_mask, 1, np.nan), cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        if cube_mask.any():
            ax.imshow(np.where(cube_mask, 1, np.nan), cmap='Oranges', alpha=0.6, vmin=0, vmax=1)
        
        ax.set_title(f"{combo['name']}\n(Middle slice Z={z_slice})", fontsize=12)
        ax.axis('off')
        
        # Add stats text
        unique_labels = np.unique(mask)
        counts = np.bincount(mask.flatten())
        stats_text = []
        label_names = ['BG', 'Mem', 'Sph', 'Cube']
        for j, count in enumerate(counts[:len(unique_labels)]):
            if j < len(label_names):
                pct = (count / mask.size) * 100
                stats_text.append(f"{label_names[j]}: {pct:.1f}%")
        
        ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
    
    plt.suptitle('Synthetic EM Volume Examples: All Structure Combinations\n(Blue=Membrane, Red=Sphere, Orange=Cube)', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('structure_combinations_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Static examples saved as structure_combinations_examples.png")
    return 'structure_combinations_examples.png'

def generate_test_volumes(n_volumes=12):
    """Generate test volumes with masks using equal combinations"""
    print("Generating test volumes with random combinations...")
    gen = MembraneGen(generate_masks=True, equal_combinations=True)  # Random combinations
    
    volumes = []
    masks = []
    seeds = [42, 123, 456, 789, 1000, 1337, 2000, 2500, 3000, 3500, 4000, 4500][:n_volumes]
    
    for i, seed in enumerate(seeds):
        print(f"  Generating volume {i+1}/{n_volumes} (seed {seed})")
        vol_bytes, mask_bytes = gen(seed)
        
        volume = np.frombuffer(vol_bytes, dtype=np.float32).reshape(96, 96, 96)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
        
        volumes.append(volume)
        masks.append(mask)
        
        # Print stats with structure types
        unique_labels = np.unique(mask)
        counts = np.bincount(mask.flatten())
        structures = []
        if 1 in unique_labels: structures.append('Mem')
        if 2 in unique_labels: structures.append('Sph')  
        if 3 in unique_labels: structures.append('Cube')
        if not structures: structures = ['BG Only']
        print(f"    Labels: {unique_labels}, Structures: {', '.join(structures)}")
    
    return volumes, masks, seeds

def volume_to_base64_images(volume, mask):
    """Convert volume and mask slices to base64 encoded images"""
    print("Converting volume to base64 images...")
    
    volume_images = []
    mask_images = []
    overlay_images = []
    
    # Normalize volume for display
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min())
    
    # Create colormap for masks
    colors = ['black', 'blue', 'red', 'orange']  # bg, membrane, sphere, cube
    cmap = mcolors.ListedColormap(colors)
    
    for z in range(volume.shape[0]):
        # Volume slice
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(vol_norm[z], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f'Volume Slice {z}')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        vol_img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        volume_images.append(vol_img_b64)
        plt.close()
        
        # Mask slice
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(mask[z], cmap=cmap, vmin=0, vmax=3)
        ax.axis('off')
        ax.set_title(f'Mask Slice {z} (Blue=Mem, Red=Sphere, Orange=Cube)')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        mask_img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        mask_images.append(mask_img_b64)
        plt.close()
        
        # Overlay slice
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(vol_norm[z], cmap='gray', vmin=0, vmax=1, alpha=0.7)
        
        # Overlay shapes with transparency
        membrane_mask = mask[z] == 1
        sphere_mask = mask[z] == 2
        cube_mask = mask[z] == 3
        
        if membrane_mask.any():
            ax.imshow(np.where(membrane_mask, 1, np.nan), cmap='Blues', alpha=0.5, vmin=0, vmax=1)
        if sphere_mask.any():
            ax.imshow(np.where(sphere_mask, 1, np.nan), cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        if cube_mask.any():
            ax.imshow(np.where(cube_mask, 1, np.nan), cmap='Oranges', alpha=0.5, vmin=0, vmax=1)
            
        ax.axis('off')
        ax.set_title(f'Overlay Slice {z}')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        overlay_img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        overlay_images.append(overlay_img_b64)
        plt.close()
        
        if (z + 1) % 20 == 0:
            print(f"    Processed {z + 1}/96 slices")
    
    return volume_images, mask_images, overlay_images

def create_interactive_html(volumes, masks, seeds, output_file="shape_viewer.html"):
    """Create interactive HTML viewer"""
    print(f"Creating interactive HTML viewer: {output_file}")
    
    # Convert all volumes to base64 images
    all_data = []
    for i, (volume, mask, seed) in enumerate(zip(volumes, masks, seeds)):
        print(f"Processing volume {i+1}/{len(volumes)}...")
        vol_imgs, mask_imgs, overlay_imgs = volume_to_base64_images(volume, mask)
        
        # Calculate statistics
        unique_labels = np.unique(mask)
        counts = np.bincount(mask.flatten())
        stats = {
            'seed': seed,
            'labels': unique_labels.tolist(),
            'counts': counts[:len(unique_labels)].tolist(),
            'volume_range': [float(volume.min()), float(volume.max())],
            'total_voxels': int(mask.size)
        }
        
        all_data.append({
            'volume_images': vol_imgs,
            'mask_images': mask_imgs,
            'overlay_images': overlay_imgs,
            'stats': stats
        })
    
    # Create HTML template
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Volume Shape Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .controls {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .slider {
            width: 300px;
        }
        .images {
            display: flex;
            justify-content: space-around;
            gap: 20px;
            margin-bottom: 20px;
        }
        .image-container {
            text-align: center;
            flex: 1;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 5px;
        }
        .stats {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .legend {
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .legend-item {
            display: inline-block;
            margin-right: 20px;
        }
        .color-box {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
            border: 1px solid #000;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>3D Volume Shape Viewer</h1>
            <p>Interactive viewer for synthetic EM volumes with membranes, spheres, and cubes</p>
        </div>
        
        <div class="legend">
            <strong>Shape Legend:</strong>
            <div class="legend-item">
                <span class="color-box" style="background-color: black;"></span>
                Background
            </div>
            <div class="legend-item">
                <span class="color-box" style="background-color: blue;"></span>
                Membrane
            </div>
            <div class="legend-item">
                <span class="color-box" style="background-color: red;"></span>
                Sphere
            </div>
            <div class="legend-item">
                <span class="color-box" style="background-color: orange;"></span>
                Cube
            </div>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="volumeSelect">Volume:</label>
                <select id="volumeSelect" onchange="changeVolume()">
                    <!-- Options will be populated by JavaScript -->
                </select>
            </div>
            
            <div class="control-group">
                <label for="sliceSlider">Slice (Z):</label>
                <input type="range" id="sliceSlider" class="slider" min="0" max="95" value="48" 
                       oninput="updateSlice(this.value)">
                <span id="sliceValue">48</span>
            </div>
            
            <div class="control-group">
                <label for="viewMode">View:</label>
                <select id="viewMode" onchange="changeViewMode()">
                    <option value="volume">Volume Only</option>
                    <option value="mask">Mask Only</option>
                    <option value="overlay" selected>Overlay</option>
                </select>
            </div>
        </div>
        
        <div class="images">
            <div class="image-container">
                <img id="currentImage" src="" alt="Current slice">
                <p id="imageCaption">Slice 48</p>
            </div>
        </div>
        
        <div class="stats">
            <h3>Volume Statistics</h3>
            <div class="stats-grid">
                <div><strong>Seed:</strong> <span id="statSeed">-</span></div>
                <div><strong>Volume Range:</strong> <span id="statRange">-</span></div>
                <div><strong>Total Voxels:</strong> <span id="statTotal">-</span></div>
                <div><strong>Background:</strong> <span id="statBg">-</span></div>
                <div><strong>Membrane:</strong> <span id="statMem">-</span></div>
                <div><strong>Spheres:</strong> <span id="statSph">-</span></div>
                <div><strong>Cubes:</strong> <span id="statCube">-</span></div>
            </div>
        </div>
    </div>

    <script>
        // Data will be injected here
        const volumeData = """ + json.dumps(all_data) + """;
        
        let currentVolume = 0;
        let currentSlice = 48;
        let currentViewMode = 'overlay';
        
        function initializeViewer() {
            // Populate volume selector
            const volumeSelect = document.getElementById('volumeSelect');
            volumeData.forEach((vol, i) => {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `Volume ${i + 1} (seed ${vol.stats.seed})`;
                volumeSelect.appendChild(option);
            });
            
            updateDisplay();
        }
        
        function changeVolume() {
            currentVolume = parseInt(document.getElementById('volumeSelect').value);
            updateDisplay();
        }
        
        function updateSlice(slice) {
            currentSlice = parseInt(slice);
            document.getElementById('sliceValue').textContent = slice;
            updateDisplay();
        }
        
        function changeViewMode() {
            currentViewMode = document.getElementById('viewMode').value;
            updateDisplay();
        }
        
        function updateDisplay() {
            const vol = volumeData[currentVolume];
            let imageData;
            
            switch(currentViewMode) {
                case 'volume':
                    imageData = vol.volume_images[currentSlice];
                    break;
                case 'mask':
                    imageData = vol.mask_images[currentSlice];
                    break;
                case 'overlay':
                default:
                    imageData = vol.overlay_images[currentSlice];
                    break;
            }
            
            document.getElementById('currentImage').src = 'data:image/png;base64,' + imageData;
            document.getElementById('imageCaption').textContent = 
                `Slice ${currentSlice} - ${currentViewMode.charAt(0).toUpperCase() + currentViewMode.slice(1)} View`;
            
            // Update stats
            const stats = vol.stats;
            document.getElementById('statSeed').textContent = stats.seed;
            document.getElementById('statRange').textContent = 
                `[${stats.volume_range[0].toFixed(3)}, ${stats.volume_range[1].toFixed(3)}]`;
            document.getElementById('statTotal').textContent = stats.total_voxels.toLocaleString();
            
            // Update shape counts
            const labels = ['statBg', 'statMem', 'statSph', 'statCube'];
            labels.forEach((id, i) => {
                const element = document.getElementById(id);
                if (i < stats.counts.length) {
                    const count = stats.counts[i];
                    const percent = ((count / stats.total_voxels) * 100).toFixed(1);
                    element.textContent = `${count.toLocaleString()} (${percent}%)`;
                } else {
                    element.textContent = '0 (0.0%)';
                }
            });
        }
        
        // Keyboard controls
        document.addEventListener('keydown', function(event) {
            if (event.key === 'ArrowUp' && currentSlice < 95) {
                currentSlice++;
                document.getElementById('sliceSlider').value = currentSlice;
                updateSlice(currentSlice);
            } else if (event.key === 'ArrowDown' && currentSlice > 0) {
                currentSlice--;
                document.getElementById('sliceSlider').value = currentSlice;
                updateSlice(currentSlice);
            }
        });
        
        // Initialize on page load
        window.onload = initializeViewer;
    </script>
</body>
</html>
"""
    
    # Save HTML file
    with open(output_file, 'w') as f:
        f.write(html_template)
    
    print(f"✓ Interactive viewer saved as {output_file}")
    print(f"  File size: {len(html_template) / 1024 / 1024:.1f} MB")
    return output_file

def main():
    """Main function to generate volumes and create viewer"""
    print("=== 3D Shape Viewer Generator ===")
    
    # Generate test volumes
    volumes, masks, seeds = generate_test_volumes(n_volumes=12)
    
    # Create interactive HTML viewer
    output_file = create_interactive_html(volumes, masks, seeds, "shape_viewer.html")
    
    print(f"\n✓ Complete! Open {output_file} in your web browser to view the interactive visualization.")
    print("\nFeatures:")
    print("  - Scroll through Z slices with slider or arrow keys")
    print("  - Switch between volumes")
    print("  - Toggle between volume, mask, and overlay views")
    print("  - View statistics for each volume")

if __name__ == "__main__":
    main() 