#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from membrane_synthetic_data_custom_intensities import MembraneSyntheticDatasetCustomIntensities
import torch

def generate_membrane_volume(seed=42):
    """
    Generate a single membrane volume with custom intensity levels.
    
    Returns:
        torch.Tensor: Generated volume with shape (1, 64, 64, 64)
    """
    dataset = MembraneSyntheticDatasetCustomIntensities(
        volume_size=(64, 64, 64),
        num_gaussians_range=(4, 6),
        gaussian_sigma_range=(20, 25),
        isovalue=0.8,
        isoband_width=0.1,
        noise_level=0.02,
        num_samples=1,
        seed=seed,
        # Sphere parameters (user's settings)
        num_additional_spheres_range=(6, 6),
        additional_sphere_radius_range=(8.0, 8.0),
        # Realism parameters
        blur_sigma=1.0,
        isovalue_variation=0.3,
        intensity_gradient_strength=0.2,
        # Custom intensity levels
        background_intensity=0.72,   # ~0.7 light background
        membrane_intensity=0.22,     # ~0.25 dark membranes
        sphere_intensity=0.03        # ~0.05 very dark spheres
    )
    
    return dataset[0]

def visualize_volume(volume, output_file="membrane_volume.png"):
    """
    Create a simple visualization of the 3D volume showing slices.
    """
    volume_np = volume.squeeze().numpy()
    D, H, W = volume_np.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Custom Intensity Membrane Volume\nShape: {D}×{H}×{W}', fontsize=16)
    
    # Top row: Different axis slices
    slice_positions = [D//2, H//2, W//2]
    axis_names = ['Z-axis (Depth)', 'Y-axis (Height)', 'X-axis (Width)']
    
    for i in range(3):
        if i == 0:  # Z-axis
            slice_data = volume_np[slice_positions[i], :, :]
        elif i == 1:  # Y-axis
            slice_data = volume_np[:, slice_positions[i], :]
        else:  # X-axis
            slice_data = volume_np[:, :, slice_positions[i]]
        
        axes[0, i].imshow(slice_data, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'{axis_names[i]}\nSlice {slice_positions[i]}')
        axes[0, i].axis('off')
    
    # Bottom row: Different Z-slices
    z_slices = [D//4, D//2, 3*D//4]
    for i, z_pos in enumerate(z_slices):
        axes[1, i].imshow(volume_np[z_pos, :, :], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Z-slice {z_pos}')
        axes[1, i].axis('off')
    
    # Add stats text
    stats_text = f"Min: {volume_np.min():.3f}\nMax: {volume_np.max():.3f}\nMean: {volume_np.mean():.3f}"
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    return fig

def main():
    """
    Generate and visualize a custom intensity membrane volume.
    """
    print("Generating custom intensity membrane volume...")
    
    # Generate volume
    volume = generate_membrane_volume(seed=42)
    
    print(f"Generated volume shape: {volume.shape}")
    print(f"Volume range: {volume.min():.4f} - {volume.max():.4f}")
    print(f"Volume mean: {volume.mean():.4f}")
    
    # Visualize
    visualize_volume(volume)
    
    print("Done!")
    return volume

if __name__ == "__main__":
    volume = main() 