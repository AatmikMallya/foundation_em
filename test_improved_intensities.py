#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from membrane_synthetic_data_custom_intensities import create_custom_intensity_membrane_dataset
from create_interactive_html_viewer import create_html_viewer
import torch

def create_improved_intensity_membrane():
    """
    Create membrane with improved intensity targeting.
    """
    print("🎯 Creating membrane with improved intensity targeting...")
    print("Target intensities:")
    print("  • Background: ~0.7 (light)")
    print("  • Membrane: ~0.25 (dark)")  
    print("  • Spheres: ~0.05 (very dark)")
    
    # Adjusted parameters to better hit targets
    dataset = create_custom_intensity_membrane_dataset(
        volume_size=(64, 64, 64),
        num_gaussians_range=(4, 6),
        gaussian_sigma_range=(20, 25),
        isovalue=0.8,
        isoband_width=0.1,
        noise_level=0.02,  # Reduced noise to maintain target intensities
        num_samples=1,
        seed=42,
        # Sphere parameters
        num_additional_spheres_range=(6, 6),
        additional_sphere_radius_range=(8.0, 8.0),
        # Realism parameters
        blur_sigma=1.0,  # Reduced blur to maintain sharper intensity differences
        isovalue_variation=0.3,  # Reduced variation
        intensity_gradient_strength=0.2,  # Reduced to maintain target intensity
        # IMPROVED INTENSITY LEVELS
        background_intensity=0.72,   # Slightly higher to account for blur/noise
        membrane_intensity=0.22,     # Lower to hit 0.25 target after blur/noise
        sphere_intensity=0.03        # Lower to hit 0.05 target after blur/noise
    )
    
    volume = dataset[0]
    print(f"Generated volume shape: {volume.shape}")
    
    return volume

def analyze_and_visualize(volume, title_suffix=""):
    """
    Analyze intensity regions and create visualizations.
    """
    volume_np = volume.squeeze().numpy() if isinstance(volume, torch.Tensor) else volume.squeeze()
    
    # Define intensity thresholds to separate regions
    background_threshold = 0.5
    sphere_threshold = 0.15
    
    # Create masks for different regions
    background_mask = volume_np > background_threshold
    membrane_mask = (volume_np >= sphere_threshold) & (volume_np <= background_threshold)
    sphere_mask = volume_np < sphere_threshold
    
    # Calculate statistics
    stats = {}
    if np.any(background_mask):
        stats['background'] = {
            'mean': volume_np[background_mask].mean(),
            'std': volume_np[background_mask].std(),
            'percentage': np.sum(background_mask) / volume_np.size * 100
        }
    
    if np.any(membrane_mask):
        stats['membrane'] = {
            'mean': volume_np[membrane_mask].mean(),
            'std': volume_np[membrane_mask].std(),
            'percentage': np.sum(membrane_mask) / volume_np.size * 100
        }
    
    if np.any(sphere_mask):
        stats['sphere'] = {
            'mean': volume_np[sphere_mask].mean(),
            'std': volume_np[sphere_mask].std(),
            'percentage': np.sum(sphere_mask) / volume_np.size * 100
        }
    
    # Print results
    print(f"\n" + "="*60)
    print(f"INTENSITY ANALYSIS{title_suffix}")
    print("="*60)
    
    target_values = {'background': 0.7, 'membrane': 0.25, 'sphere': 0.05}
    
    for region in ['background', 'membrane', 'sphere']:
        if region in stats:
            achieved = stats[region]['mean']
            target = target_values[region]
            diff = achieved - target
            status = "✅" if abs(diff) < 0.05 else "⚠️" if abs(diff) < 0.1 else "❌"
            
            print(f"\n{region.upper()} REGION {status}")
            print(f"  Achieved: {achieved:.4f}")
            print(f"  Target: {target:.4f}")
            print(f"  Difference: {diff:+.4f}")
            print(f"  Volume: {stats[region]['percentage']:.1f}%")
    
    overall_mean = volume_np.mean()
    print(f"\nOVERALL VOLUME:")
    print(f"  Mean: {overall_mean:.4f}")
    print(f"  Min: {volume_np.min():.4f}, Max: {volume_np.max():.4f}")
    
    return stats

def create_side_by_side_comparison():
    """
    Create a side-by-side comparison of original vs improved intensities.
    """
    print("Creating side-by-side comparison...")
    
    # Original parameters (from first test)
    dataset_original = create_custom_intensity_membrane_dataset(
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
        background_intensity=0.7,
        membrane_intensity=0.25,
        sphere_intensity=0.05
    )
    
    # Improved parameters
    dataset_improved = create_custom_intensity_membrane_dataset(
        volume_size=(64, 64, 64),
        num_gaussians_range=(4, 6),
        gaussian_sigma_range=(20, 25),
        isovalue=0.8,
        isoband_width=0.1,
        noise_level=0.015,
        num_samples=1,
        seed=42,
        num_additional_spheres_range=(2, 5),
        additional_sphere_radius_range=(3.0, 8.0),
        blur_sigma=1.0,
        isovalue_variation=0.3,
        intensity_gradient_strength=0.2,
        background_intensity=0.72,
        membrane_intensity=0.22,
        sphere_intensity=0.03
    )
    
    vol_original = dataset_original[0]
    vol_improved = dataset_improved[0]
    
    print("\n🔍 ORIGINAL INTENSITIES:")
    stats_orig = analyze_and_visualize(vol_original, " - ORIGINAL")
    
    print("\n🎯 IMPROVED INTENSITIES:")
    stats_improved = analyze_and_visualize(vol_improved, " - IMPROVED")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Intensity Comparison: Original vs Improved', fontsize=16)
    
    for row, (volume, title) in enumerate([(vol_original, "Original"), (vol_improved, "Improved")]):
        volume_np = volume.squeeze().numpy()
        D, H, W = volume_np.shape
        
        # Show middle slices from different axes
        slices = [
            volume_np[D//2, :, :],      # Z-axis
            volume_np[:, H//2, :],      # Y-axis  
            volume_np[:, :, W//2],      # X-axis
        ]
        
        for col in range(3):
            axes[row, col].imshow(slices[col], cmap='gray', vmin=0, vmax=1)
            axes[row, col].set_title(f'{title} - {"ZYX"[col]}-axis slice')
            axes[row, col].axis('off')
        
        # Histogram comparison
        axes[row, 3].hist(volume_np.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[row, 3].axvline(volume_np.mean(), color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {volume_np.mean():.3f}')
        axes[row, 3].set_xlabel('Intensity')
        axes[row, 3].set_ylabel('Frequency')
        axes[row, 3].set_title(f'{title} - Intensity Distribution')
        axes[row, 3].legend()
        axes[row, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('intensity_comparison_original_vs_improved.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison visualization saved to: intensity_comparison_original_vs_improved.png")
    
    # Create interactive HTML for improved version
    html_file = create_html_viewer(vol_improved, "improved_intensity_membrane_viewer.html")
    print(f"Interactive HTML viewer created: {html_file}")
    
    return vol_improved, stats_improved

if __name__ == "__main__":
    print("🎛️ Testing Improved Custom Intensity Membrane Generator")
    
    # Create and analyze improved version
    improved_volume, improved_stats = create_side_by_side_comparison()
    
    print(f"\n✅ SUCCESS! Files created:")
    print(f"  • intensity_comparison_original_vs_improved.png - Side-by-side comparison")
    print(f"  • improved_intensity_membrane_viewer.html - Interactive HTML viewer")
    
    print(f"\n🎯 FINAL ACHIEVED INTENSITY LEVELS:")
    target_values = {'background': 0.7, 'membrane': 0.25, 'sphere': 0.05}
    for region in ['background', 'membrane', 'sphere']:
        if region in improved_stats:
            achieved = improved_stats[region]['mean']
            target = target_values[region]
            error = abs(achieved - target)
            status = "✅" if error < 0.05 else "⚠️" if error < 0.1 else "❌"
            print(f"  {status} {region.title()}: {achieved:.3f} (target: {target}, error: {error:.3f})") 