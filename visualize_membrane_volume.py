#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from membrane_synthetic_data import MembraneSyntheticDataset
import torch

def visualize_3d_volume(volume, title="3D Volume", num_slices=9):
    """
    Visualize a 3D volume by showing slices along different axes.
    """
    volume_np = volume.squeeze().numpy() if isinstance(volume, torch.Tensor) else volume.squeeze()
    D, H, W = volume_np.shape
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)
    
    # Z-axis slices (depth)
    for i in range(3):
        slice_idx = int((i + 1) * D / 4)
        axes[0, i].imshow(volume_np[slice_idx, :, :], cmap='gray')
        axes[0, i].set_title(f'Z-slice {slice_idx}')
        axes[0, i].axis('off')
    
    # Y-axis slices (height)
    for i in range(3):
        slice_idx = int((i + 1) * H / 4)
        axes[1, i].imshow(volume_np[:, slice_idx, :], cmap='gray')
        axes[1, i].set_title(f'Y-slice {slice_idx}')
        axes[1, i].axis('off')
    
    # X-axis slices (width)
    for i in range(3):
        slice_idx = int((i + 1) * W / 4)
        axes[2, i].imshow(volume_np[:, :, slice_idx], cmap='gray')
        axes[2, i].set_title(f'X-slice {slice_idx}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    return fig

def generate_and_visualize_membrane():
    """
    Generate a synthetic membrane volume and visualize it.
    """
    print("Creating MembraneSyntheticDataset...")
    
    # Create dataset with interesting parameters
    dataset = MembraneSyntheticDataset(
        volume_size=(64, 64, 64),
        num_gaussians_range=(4, 6),  # Fewer Gaussians = larger continuous structures
        gaussian_sigma_range=(20, 25),  # Much larger sigma = wider Gaussians = larger membranes
        isovalue=0.8,
        isoband_width=0.1,
        noise_level=0.02,
        num_samples=1,
        seed=42,
        # Add some spheres for more interesting structure
        num_additional_spheres_range=(2, 5),  # Reduced sphere count
        additional_sphere_radius_range=(3.0, 8.0),  # Larger spheres
        # Enhanced realism parameters
        blur_sigma=1.5,  # Slightly more blur for smoother large structures
        isovalue_variation=0.5,
        intensity_gradient_strength=0.4
    )
    
    print("Generating volume...")
    volume = dataset[0]  # Generate first volume
    
    print(f"Generated volume shape: {volume.shape}")
    print(f"Volume data type: {volume.dtype}")
    print(f"Volume min value: {volume.min():.4f}")
    print(f"Volume max value: {volume.max():.4f}")
    print(f"Volume mean value: {volume.mean():.4f}")
    
    # Create visualization
    fig = visualize_3d_volume(volume, "Synthetic Membrane Structure")
    
    # Save the visualization
    output_file = "membrane_volume_visualization.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    # Also create a histogram of intensity values
    plt.figure(figsize=(10, 6))
    volume_np = volume.squeeze().numpy()
    plt.hist(volume_np.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Intensity Values in Synthetic Membrane')
    plt.grid(True, alpha=0.3)
    
    histogram_file = "membrane_intensity_histogram.png"
    plt.savefig(histogram_file, dpi=150, bbox_inches='tight')
    print(f"Intensity histogram saved to {histogram_file}")
    
    plt.show()

if __name__ == "__main__":
    generate_and_visualize_membrane() 