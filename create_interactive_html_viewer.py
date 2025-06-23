#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from membrane_synthetic_data import MembraneSyntheticDataset
import torch
import base64
import io
from datetime import datetime

def volume_to_base64_images(volume, axis=0):
    """
    Convert volume slices to base64 encoded images for embedding in HTML.
    
    Args:
        volume: 3D numpy array
        axis: 0=Z, 1=Y, 2=X
    
    Returns:
        List of base64 encoded image strings
    """
    images = []
    
    if axis == 0:  # Z-axis slices
        num_slices = volume.shape[0]
        slice_func = lambda i: volume[i, :, :]
    elif axis == 1:  # Y-axis slices
        num_slices = volume.shape[1]
        slice_func = lambda i: volume[:, i, :]
    else:  # X-axis slices
        num_slices = volume.shape[2]
        slice_func = lambda i: volume[:, :, i]
    
    for i in range(num_slices):
        # Create matplotlib figure for this slice
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(slice_func(i), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f'Slice {i}')
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        images.append(image_base64)
        plt.close(fig)
    
    return images

def create_html_viewer(volume, output_filename="interactive_membrane_viewer.html"):
    """
    Create an interactive HTML file for viewing 3D volume slices.
    """
    volume_np = volume.squeeze().numpy() if isinstance(volume, torch.Tensor) else volume.squeeze()
    D, H, W = volume_np.shape
    
    print("Converting slices to images...")
    
    # Generate images for all three axes
    z_images = volume_to_base64_images(volume_np, axis=0)
    y_images = volume_to_base64_images(volume_np, axis=1)
    x_images = volume_to_base64_images(volume_np, axis=2)
    
    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive 3D Membrane Volume Viewer</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .stats {{
            background-color: #e8f4f8;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        .controls {{
            margin: 20px 0;
            text-align: center;
        }}
        .axis-buttons {{
            margin: 10px 0;
        }}
        .axis-button {{
            padding: 8px 16px;
            margin: 0 5px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        .axis-button.active {{
            background-color: #28a745;
        }}
        .axis-button:hover {{
            opacity: 0.8;
        }}
        .slider-container {{
            margin: 20px 0;
        }}
        .slider {{
            width: 100%;
            height: 20px;
            -webkit-appearance: none;
            appearance: none;
            background: #ddd;
            outline: none;
            border-radius: 10px;
        }}
        .slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            background: #007bff;
            cursor: pointer;
            border-radius: 50%;
        }}
        .slider::-moz-range-thumb {{
            width: 20px;
            height: 20px;
            background: #007bff;
            cursor: pointer;
            border-radius: 50%;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .volume-image {{
            max-width: 100%;
            border: 2px solid #ddd;
            border-radius: 5px;
        }}
        .slice-info {{
            margin: 10px 0;
            font-weight: bold;
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Interactive 3D Membrane Volume Viewer</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <strong>Volume Statistics:</strong><br>
            Shape: {D} × {H} × {W}<br>
            Min: {volume_np.min():.4f}, Max: {volume_np.max():.4f}<br>
            Mean: {volume_np.mean():.4f}, Std: {volume_np.std():.4f}
        </div>
        
        <div class="controls">
            <div class="axis-buttons">
                <button class="axis-button active" onclick="switchAxis(0)">Z-axis (Depth) - {D} slices</button>
                <button class="axis-button" onclick="switchAxis(1)">Y-axis (Height) - {H} slices</button>
                <button class="axis-button" onclick="switchAxis(2)">X-axis (Width) - {W} slices</button>
            </div>
            
            <div class="slider-container">
                <input type="range" class="slider" id="sliceSlider" min="0" max="{D-1}" value="{D//2}" oninput="updateSlice(this.value)">
            </div>
            
            <div class="slice-info" id="sliceInfo">
                Z-axis - Slice {D//2} / {D-1}
            </div>
        </div>
        
        <div class="image-container">
            <img id="volumeImage" class="volume-image" src="data:image/png;base64,{z_images[D//2]}" alt="Volume Slice">
        </div>
        
        <div style="text-align: center; margin-top: 20px; font-size: 12px; color: #666;">
            🎛️ Use the slider to scroll through slices • Click axis buttons to change viewing direction
        </div>
    </div>

    <script>
        // Store all images
        const imageData = {{
            0: {z_images},  // Z-axis images
            1: {y_images},  // Y-axis images  
            2: {x_images}   // X-axis images
        }};
        
        const axisNames = ['Z-axis (Depth)', 'Y-axis (Height)', 'X-axis (Width)'];
        const maxSlices = [{D-1}, {H-1}, {W-1}];
        
        let currentAxis = 0;
        let currentSlice = {D//2};
        
        function updateSlice(sliceIndex) {{
            currentSlice = parseInt(sliceIndex);
            const image = document.getElementById('volumeImage');
            const sliceInfo = document.getElementById('sliceInfo');
            
            image.src = `data:image/png;base64,${{imageData[currentAxis][currentSlice]}}`;
            sliceInfo.textContent = `${{axisNames[currentAxis]}} - Slice ${{currentSlice}} / ${{maxSlices[currentAxis]}}`;
        }}
        
        function switchAxis(axis) {{
            currentAxis = axis;
            const slider = document.getElementById('sliceSlider');
            const buttons = document.querySelectorAll('.axis-button');
            
            // Update button styles
            buttons.forEach((btn, idx) => {{
                btn.classList.toggle('active', idx === axis);
            }});
            
            // Update slider range
            slider.max = maxSlices[axis];
            currentSlice = Math.min(currentSlice, maxSlices[axis]);
            slider.value = currentSlice;
            
            // Update image and info
            updateSlice(currentSlice);
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', function(event) {{
            const slider = document.getElementById('sliceSlider');
            const max = parseInt(slider.max);
            
            if (event.key === 'ArrowLeft' && currentSlice > 0) {{
                slider.value = currentSlice - 1;
                updateSlice(currentSlice - 1);
            }} else if (event.key === 'ArrowRight' && currentSlice < max) {{
                slider.value = currentSlice + 1;
                updateSlice(currentSlice + 1);
            }}
        }});
    </script>
</body>
</html>
    """
    
    # Write HTML file
    with open(output_filename, 'w') as f:
        f.write(html_content)
    
    print(f"Interactive HTML viewer saved to: {output_filename}")
    print(f"File size: {len(html_content) / 1024 / 1024:.1f} MB")
    return output_filename

def generate_interactive_html():
    """
    Generate membrane volume and create interactive HTML viewer.
    """
    print("Creating MembraneSyntheticDataset with your current parameters...")
    
    # Use the user's current parameters
    dataset = MembraneSyntheticDataset(
        volume_size=(64, 64, 64),
        num_gaussians_range=(4, 6),  # User's current setting
        gaussian_sigma_range=(20, 25),  # User's current setting
        isovalue=0.8,  # User's current setting
        isoband_width=0.1,  # User's current setting
        noise_level=0.02,
        num_samples=1,
        seed=42,
        num_additional_spheres_range=(2, 5),
        additional_sphere_radius_range=(3.0, 8.0),
        blur_sigma=1.5,
        isovalue_variation=0.5,  # User's current setting
        intensity_gradient_strength=0.4
    )
    
    print("Generating volume...")
    volume = dataset[0]
    
    print(f"Volume shape: {volume.shape}")
    print(f"Volume range: {volume.min():.4f} - {volume.max():.4f}")
    
    # Create interactive HTML file
    output_file = create_html_viewer(volume)
    
    print(f"\n✅ SUCCESS! Interactive viewer created: {output_file}")
    print("\n📁 To use:")
    print("1. Download this file to your local computer")
    print("2. Open it in any web browser")
    print("3. Use slider and buttons to explore the 3D volume")
    print("\n🎛️ Features:")
    print("• Slider to scroll through all slices")
    print("• Buttons to switch between Z/Y/X axes")
    print("• Keyboard arrow keys for navigation")
    print("• Works offline - everything embedded in one file")
    
    return output_file

if __name__ == "__main__":
    generate_interactive_html() 