#!/usr/bin/env python3
"""
Test intensity variation for spheres and cubes
"""

import numpy as np
import matplotlib.pyplot as plt
from vol_generator import MembraneGen

def test_intensity_variation():
    """Test current intensity variation"""
    print("Testing current intensity variation...")
    
    gen = MembraneGen(generate_masks=True, equal_combinations=False)
    
    # Generate multiple volumes and collect intensity statistics
    sphere_intensities = []
    cube_intensities = []
    
    for seed in range(50):  # Test 50 volumes
        vol_bytes, mask_bytes = gen(seed)
        volume = np.frombuffer(vol_bytes, dtype=np.float32).reshape(96, 96, 96)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
        
        # Extract sphere intensities
        sphere_voxels = volume[mask == 2]
        if len(sphere_voxels) > 0:
            sphere_intensities.extend(sphere_voxels)
            
        # Extract cube intensities  
        cube_voxels = volume[mask == 3]
        if len(cube_voxels) > 0:
            cube_intensities.extend(cube_voxels)
    
    print(f"Collected {len(sphere_intensities)} sphere voxels")
    print(f"Collected {len(cube_intensities)} cube voxels")
    
    if sphere_intensities:
        print(f"Sphere intensity range: {min(sphere_intensities):.4f} - {max(sphere_intensities):.4f}")
        print(f"Sphere intensity mean: {np.mean(sphere_intensities):.4f} ± {np.std(sphere_intensities):.4f}")
    
    if cube_intensities:
        print(f"Cube intensity range: {min(cube_intensities):.4f} - {max(cube_intensities):.4f}")
        print(f"Cube intensity mean: {np.mean(cube_intensities):.4f} ± {np.std(cube_intensities):.4f}")
    
    # Plot histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if sphere_intensities:
        ax1.hist(sphere_intensities, bins=50, alpha=0.7, color='red', label='Spheres')
        ax1.set_title('Sphere Intensity Distribution')
        ax1.set_xlabel('Intensity')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
    
    if cube_intensities:
        ax2.hist(cube_intensities, bins=50, alpha=0.7, color='orange', label='Cubes')
        ax2.set_title('Cube Intensity Distribution')
        ax2.set_xlabel('Intensity')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('current_intensity_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Current intensity distributions saved as current_intensity_distributions.png")

def test_specific_examples():
    """Generate specific examples to show intensity differences"""
    print("\nGenerating specific examples...")
    
    gen = MembraneGen(generate_masks=True, equal_combinations=False)
    
    # Force specific combinations
    sphere_combo = {'membrane': False, 'spheres': True, 'cubes': False}
    cube_combo = {'membrane': False, 'spheres': False, 'cubes': True}
    both_combo = {'membrane': False, 'spheres': True, 'cubes': True}
    
    # Generate examples
    examples = []
    for i, combo in enumerate([sphere_combo, cube_combo, both_combo]):
        gen.combinations = [combo]
        vol_bytes, mask_bytes = gen(1000 + i)
        volume = np.frombuffer(vol_bytes, dtype=np.float32).reshape(96, 96, 96)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
        examples.append((volume, mask, combo))
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i, (volume, mask, combo) in enumerate(examples):
        # Show middle slice
        z_slice = 48
        vol_norm = (volume - volume.min()) / (volume.max() - volume.min())
        
        # Volume view
        ax = axes[i, 0]
        ax.imshow(vol_norm[z_slice], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Volume (normalized)")
        ax.axis('off')
        
        # Mask view
        ax = axes[i, 1]
        ax.imshow(mask[z_slice], cmap='jet', vmin=0, vmax=3)
        ax.set_title(f"Mask")
        ax.axis('off')
        
        # Raw intensity view (not normalized)
        ax = axes[i, 2]
        im = ax.imshow(volume[z_slice], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"Raw intensities")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Print intensity stats for this volume
        if 2 in mask:  # spheres
            sphere_intensities = volume[mask == 2]
            print(f"  Example {i+1} spheres: {sphere_intensities.min():.4f} - {sphere_intensities.max():.4f} (mean: {sphere_intensities.mean():.4f})")
        
        if 3 in mask:  # cubes
            cube_intensities = volume[mask == 3]
            print(f"  Example {i+1} cubes: {cube_intensities.min():.4f} - {cube_intensities.max():.4f} (mean: {cube_intensities.mean():.4f})")
    
    # Add row labels
    row_labels = ['Spheres Only', 'Cubes Only', 'Spheres + Cubes']
    for i, label in enumerate(row_labels):
        axes[i, 0].text(-0.1, 0.5, label, transform=axes[i, 0].transAxes, 
                       rotation=90, verticalalignment='center', fontsize=12, fontweight='bold')
    
    plt.suptitle('Current Intensity Examples (Raw Values)', fontsize=16)
    plt.tight_layout()
    plt.savefig('current_intensity_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Current intensity examples saved as current_intensity_examples.png")

def main():
    print("=== Testing Current Intensity Variation ===")
    test_intensity_variation()
    test_specific_examples()
    
    print("\n=== Analysis ===")
    print("Current issues:")
    print("1. Very small per-object variation (±0.01)")
    print("2. Both spheres and cubes clipped to same range (0.01-0.15)")
    print("3. Base intensities too close (sphere: 0.03, cube: 0.05)")
    print("\nRecommendations:")
    print("1. Increase per-object variation to ±0.02 or ±0.03")
    print("2. Use different intensity ranges for spheres vs cubes")
    print("3. Increase separation between base intensities")

if __name__ == "__main__":
    main() 