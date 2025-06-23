#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

def load_voronoi_volume(bin_file_path):
    """
    Load a binary volume file (64x64x64 float32).
    
    Args:
        bin_file_path (str): Path to the .bin file
        
    Returns:
        np.ndarray: 3D volume array with shape (64, 64, 64)
    """
    # Load binary data as float32
    data = np.fromfile(bin_file_path, dtype=np.float32)
    
    # Reshape to 64x64x64
    volume = data.reshape(64, 64, 64)
    
    return volume

def visualize_voronoi_volume(volume, output_file="voronoi_volume_visualization.png"):
    """
    Create a comprehensive visualization of the 3D Voronoi volume.
    """
    D, H, W = volume.shape
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Voronoi Volume Visualization\nShape: {D}×{H}×{W}', fontsize=16)
    
    # Row 1: Different axis slices (middle slices)
    slice_positions = [D//2, H//2, W//2]
    axis_names = ['Z-axis (Depth)', 'Y-axis (Height)', 'X-axis (Width)']
    
    for i in range(3):
        if i == 0:  # Z-axis
            slice_data = volume[slice_positions[i], :, :]
        elif i == 1:  # Y-axis
            slice_data = volume[:, slice_positions[i], :]
        else:  # X-axis
            slice_data = volume[:, :, slice_positions[i]]
        
        im = axes[0, i].imshow(slice_data, cmap='viridis')
        axes[0, i].set_title(f'{axis_names[i]}\nSlice {slice_positions[i]}')
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], shrink=0.8)
    
    # Histogram
    axes[0, 3].hist(volume.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 3].axvline(volume.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {volume.mean():.3f}')
    axes[0, 3].set_xlabel('Value')
    axes[0, 3].set_ylabel('Frequency')
    axes[0, 3].set_title('Value Distribution')
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)
    
    # Row 2: Different Z-slices (depth progression)
    z_slices = [D//8, D//4, D//2, 3*D//4]
    for i, z_pos in enumerate(z_slices):
        im = axes[1, i].imshow(volume[z_pos, :, :], cmap='viridis')
        axes[1, i].set_title(f'Z-slice {z_pos}')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], shrink=0.8)
    
    # Row 3: Different Y-slices (height progression)
    y_slices = [H//8, H//4, H//2, 3*H//4]
    for i, y_pos in enumerate(y_slices):
        im = axes[2, i].imshow(volume[:, y_pos, :], cmap='viridis')
        axes[2, i].set_title(f'Y-slice {y_pos}')
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], shrink=0.8)
    
    # Add volume statistics
    stats_text = f"""Volume Statistics:
Min: {volume.min():.4f}
Max: {volume.max():.4f}
Mean: {volume.mean():.4f}
Std: {volume.std():.4f}
Shape: {volume.shape}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"images/{output_file}", dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: images/{output_file}")
    
    return fig

def extract_and_visualize_first_volume():
    """
    Extract and visualize the first volume from the first shard.
    """
    print("Extracting first volume from voronoi_volumes_64 shard_00000...")
    
    shard_path = "/gpfs/radev/scratch/clark_damon/am3833/voronoi_volumes_64/shard_00000.tar"
    temp_dir = "/tmp"
    bin_file = "v00000_00000.bin"
    temp_bin_path = os.path.join(temp_dir, bin_file)
    
    # Extract the first volume if not already extracted
    if not os.path.exists(temp_bin_path):
        import subprocess
        print(f"Extracting {bin_file} from shard...")
        result = subprocess.run([
            "tar", "-xf", shard_path, "-C", temp_dir, bin_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error extracting: {result.stderr}")
            return None
    
    print(f"Loading volume from {temp_bin_path}...")
    volume = load_voronoi_volume(temp_bin_path)
    
    print(f"Loaded volume shape: {volume.shape}")
    print(f"Volume range: {volume.min():.4f} - {volume.max():.4f}")
    print(f"Volume mean: {volume.mean():.4f}")
    
    # Visualize
    print("Creating visualization...")
    visualize_voronoi_volume(volume, "voronoi_first_volume_visualization.png")
    
    # Clean up
    if os.path.exists(temp_bin_path):
        os.remove(temp_bin_path)
        print("Cleaned up temporary file")
    
    print("Done!")
    return volume

if __name__ == "__main__":
    volume = extract_and_visualize_first_volume() 