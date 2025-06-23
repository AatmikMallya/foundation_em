#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from membrane_synthetic_data_custom_intensities import create_custom_intensity_membrane_dataset
import torch

def analyze_intensity_regions(volume, show_histogram=True):
    """
    Analyze and display intensity statistics for different regions.
    """
    volume_np = volume.squeeze().numpy() if isinstance(volume, torch.Tensor) else volume.squeeze()
    
    # Define intensity thresholds to separate regions
    background_threshold = 0.5  # Above this is likely background
    sphere_threshold = 0.15     # Below this is likely sphere
    
    # Create masks for different regions
    background_mask = volume_np > background_threshold
    membrane_mask = (volume_np >= sphere_threshold) & (volume_np <= background_threshold)
    sphere_mask = volume_np < sphere_threshold
    
    # Calculate statistics for each region
    stats = {}
    if np.any(background_mask):
        stats['background'] = {
            'mean': volume_np[background_mask].mean(),
            'std': volume_np[background_mask].std(),
            'count': np.sum(background_mask),
            'percentage': np.sum(background_mask) / volume_np.size * 100
        }
    
    if np.any(membrane_mask):
        stats['membrane'] = {
            'mean': volume_np[membrane_mask].mean(),
            'std': volume_np[membrane_mask].std(),
            'count': np.sum(membrane_mask),
            'percentage': np.sum(membrane_mask) / volume_np.size * 100
        }
    
    if np.any(sphere_mask):
        stats['sphere'] = {
            'mean': volume_np[sphere_mask].mean(),
            'std': volume_np[sphere_mask].std(),
            'count': np.sum(sphere_mask),
            'percentage': np.sum(sphere_mask) / volume_np.size * 100
        }
    
    # Print statistics
    print("\n" + "="*60)
    print("INTENSITY ANALYSIS BY REGION")
    print("="*60)
    
    for region, data in stats.items():
        print(f"\n{region.upper()} REGION:")
        print(f"  Mean intensity: {data['mean']:.4f}")
        print(f"  Std deviation: {data['std']:.4f}")
        print(f"  Volume percentage: {data['percentage']:.1f}%")
        print(f"  Voxel count: {data['count']:,}")
    
    overall_mean = volume_np.mean()
    overall_std = volume_np.std()
    print(f"\nOVERALL VOLUME:")
    print(f"  Mean intensity: {overall_mean:.4f}")
    print(f"  Std deviation: {overall_std:.4f}")
    print(f"  Min: {volume_np.min():.4f}, Max: {volume_np.max():.4f}")
    
    if show_histogram:
        # Create histogram
        plt.figure(figsize=(12, 8))
        
        # Plot overall histogram
        plt.subplot(2, 2, 1)
        plt.hist(volume_np.flatten(), bins=50, alpha=0.7, color='gray', edgecolor='black')
        plt.axvline(overall_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {overall_mean:.3f}')
        plt.xlabel('Intensity Value')
        plt.ylabel('Frequency')
        plt.title('Overall Intensity Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot histograms by region
        colors = {'background': 'lightblue', 'membrane': 'orange', 'sphere': 'darkred'}
        for i, (region, data) in enumerate(stats.items(), 2):
            plt.subplot(2, 2, i)
            region_values = volume_np[eval(f"{region}_mask")]
            plt.hist(region_values, bins=30, alpha=0.7, color=colors[region], edgecolor='black')
            plt.axvline(data['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {data["mean"]:.3f}')
            plt.xlabel('Intensity Value')
            plt.ylabel('Frequency')
            plt.title(f'{region.title()} Region Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('custom_intensity_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\nHistogram analysis saved to: custom_intensity_analysis.png")
    
    return stats

def create_intensity_comparison():
    """
    Create a comparison showing before/after custom intensity adjustment.
    """
    print("Generating volumes with custom intensity levels...")
    
    # Create dataset with your desired intensity levels
    dataset = create_custom_intensity_membrane_dataset(
        volume_size=(64, 64, 64),
        num_gaussians_range=(4, 6),  # Large structures
        gaussian_sigma_range=(20, 25),  # Large sigma
        isovalue=0.8,
        isoband_width=0.1,
        noise_level=0.02,
        num_samples=1,
        seed=42,
        # Sphere parameters
        num_additional_spheres_range=(2, 5),
        additional_sphere_radius_range=(3.0, 8.0),
        # Realism parameters
        blur_sigma=1.5,
        isovalue_variation=0.5,
        intensity_gradient_strength=0.4,
        # CUSTOM INTENSITY LEVELS
        background_intensity=0.7,   # Light background
        membrane_intensity=0.25,    # Dark membranes  
        sphere_intensity=0.05       # Very dark spheres
    )
    
    volume = dataset[0]
    print(f"Generated volume shape: {volume.shape}")
    
    # Analyze intensity regions
    stats = analyze_intensity_regions(volume, show_histogram=True)
    
    # Create slice visualization
    volume_np = volume.squeeze().numpy()
    D, H, W = volume_np.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Custom Intensity Membrane Volume - Slice Views', fontsize=16)
    
    # Show slices from middle of each axis
    slice_positions = [D//2, H//2, W//2]
    axis_names = ['Z-axis (Depth)', 'Y-axis (Height)', 'X-axis (Width)']
    
    for axis in range(3):
        if axis == 0:  # Z-axis
            slice_data = volume_np[slice_positions[axis], :, :]
        elif axis == 1:  # Y-axis
            slice_data = volume_np[:, slice_positions[axis], :]
        else:  # X-axis
            slice_data = volume_np[:, :, slice_positions[axis]]
        
        # Original view
        axes[0, axis].imshow(slice_data, cmap='gray', vmin=0, vmax=1)
        axes[0, axis].set_title(f'{axis_names[axis]} - Slice {slice_positions[axis]}')
        axes[0, axis].axis('off')
        
        # Enhanced contrast view
        axes[1, axis].imshow(slice_data, cmap='gray', vmin=0, vmax=0.8)  # Boost contrast
        axes[1, axis].set_title(f'{axis_names[axis]} - Enhanced Contrast')
        axes[1, axis].axis('off')
    
    plt.tight_layout()
    plt.savefig('custom_intensity_membrane_slices.png', dpi=150, bbox_inches='tight')
    print(f"Slice visualization saved to: custom_intensity_membrane_slices.png")
    
    return volume, stats

def create_interactive_html_with_custom_intensities():
    """
    Create an interactive HTML viewer with the custom intensity membrane.
    """
    from create_interactive_html_viewer import create_html_viewer
    
    print("\nCreating custom intensity dataset...")
    
    # Create dataset with custom intensities
    dataset = create_custom_intensity_membrane_dataset(
        volume_size=(64, 64, 64),
        num_gaussians_range=(4, 6),
        gaussian_sigma_range=(20, 25),
        isovalue=0.8,
        isoband_width=0.1,
        noise_level=0.02,
        num_samples=1,
        seed=42,
        num_additional_spheres_range=(2, 5),
        additional_sphere_radius_range=(3.0, 8.0),
        blur_sigma=1.5,
        isovalue_variation=0.5,
        intensity_gradient_strength=0.4,
        # Custom intensities
        background_intensity=0.7,
        membrane_intensity=0.25,
        sphere_intensity=0.05
    )
    
    volume = dataset[0]
    
    # Create interactive HTML viewer
    output_file = create_html_viewer(volume, "custom_intensity_membrane_viewer.html")
    print(f"\nInteractive HTML viewer created: {output_file}")
    
    return output_file

if __name__ == "__main__":
    print("🎛️ Testing Custom Intensity Membrane Generator")
    print("Target intensities:")
    print("  • Background: ~0.7 (light)")
    print("  • Membrane: ~0.25 (dark)")
    print("  • Spheres: ~0.05 (very dark)")
    
    # Test and analyze
    volume, stats = create_intensity_comparison()
    
    # Create interactive viewer
    html_file = create_interactive_html_with_custom_intensities()
    
    print(f"\n✅ SUCCESS! Files created:")
    print(f"  • custom_intensity_analysis.png - Intensity histograms")
    print(f"  • custom_intensity_membrane_slices.png - Slice visualizations")
    print(f"  • {html_file} - Interactive HTML viewer")
    
    print(f"\n📊 ACHIEVED INTENSITY LEVELS:")
    target_values = {'background': 0.7, 'membrane': 0.25, 'sphere': 0.05}
    for region in ['background', 'membrane', 'sphere']:
        if region in stats:
            print(f"  • {region.title()}: {stats[region]['mean']:.3f} (target: {target_values[region]})") 