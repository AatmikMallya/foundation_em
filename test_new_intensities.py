#!/usr/bin/env python3
"""
Test the new improved intensity variation
"""

import numpy as np
import matplotlib.pyplot as plt
from vol_generator import MembraneGen

def test_new_intensities():
    """Test the improved intensity variation"""
    print("Testing improved intensity variation...")
    
    gen = MembraneGen(generate_masks=True, equal_combinations=False)
    
    # Generate multiple volumes and collect intensity statistics
    sphere_intensities = []
    cube_intensities = []
    bg_intensities = []
    
    for seed in range(20):  # Test 20 volumes
        vol_bytes, mask_bytes = gen(seed)
        volume = np.frombuffer(vol_bytes, dtype=np.float32).reshape(96, 96, 96)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
        
        # Extract intensities by type
        bg_voxels = volume[mask == 0]
        sphere_voxels = volume[mask == 2]
        cube_voxels = volume[mask == 3]
        
        if len(bg_voxels) > 0:
            bg_intensities.extend(bg_voxels[:1000])  # Sample to avoid too much data
        if len(sphere_voxels) > 0:
            sphere_intensities.extend(sphere_voxels)
        if len(cube_voxels) > 0:
            cube_intensities.extend(cube_voxels)
    
    print(f"Background: {len(bg_intensities)} voxels")
    print(f"Spheres: {len(sphere_intensities)} voxels") 
    print(f"Cubes: {len(cube_intensities)} voxels")
    
    if bg_intensities:
        print(f"Background range: {min(bg_intensities):.3f} - {max(bg_intensities):.3f}")
        print(f"Background mean: {np.mean(bg_intensities):.3f} ± {np.std(bg_intensities):.3f}")
    
    if sphere_intensities:
        print(f"Sphere range: {min(sphere_intensities):.3f} - {max(sphere_intensities):.3f}")
        print(f"Sphere mean: {np.mean(sphere_intensities):.3f} ± {np.std(sphere_intensities):.3f}")
    
    if cube_intensities:
        print(f"Cube range: {min(cube_intensities):.3f} - {max(cube_intensities):.3f}")
        print(f"Cube mean: {np.mean(cube_intensities):.3f} ± {np.std(cube_intensities):.3f}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Histogram comparison
    ax = axes[0]
    if bg_intensities:
        ax.hist(bg_intensities, bins=50, alpha=0.7, color='gray', label='Background', density=True)
    if sphere_intensities:
        ax.hist(sphere_intensities, bins=50, alpha=0.7, color='red', label='Spheres', density=True)
    if cube_intensities:
        ax.hist(cube_intensities, bins=50, alpha=0.7, color='orange', label='Cubes', density=True)
    ax.set_title('Intensity Distributions (Overlaid)')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Individual histograms
    ax = axes[1]
    if sphere_intensities:
        ax.hist(sphere_intensities, bins=30, alpha=0.8, color='red')
        ax.set_title('Sphere Intensities')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    if cube_intensities:
        ax.hist(cube_intensities, bins=30, alpha=0.8, color='orange')
        ax.set_title('Cube Intensities')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_intensity_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Improved intensity distributions saved as improved_intensity_distributions.png")
    
    # Calculate separation metrics
    if sphere_intensities and cube_intensities:
        sphere_mean = np.mean(sphere_intensities)
        cube_mean = np.mean(cube_intensities)
        separation = abs(cube_mean - sphere_mean)
        sphere_std = np.std(sphere_intensities)
        cube_std = np.std(cube_intensities)
        
        print(f"\n=== Separation Analysis ===")
        print(f"Mean separation: {separation:.3f}")
        print(f"Separation in std devs: {separation / np.mean([sphere_std, cube_std]):.2f}")
        print(f"Cube/Sphere intensity ratio: {cube_mean / sphere_mean:.2f}x")

def main():
    print("=== Testing Improved Intensity Parameters ===")
    print("New parameters:")
    print("- Sphere base: 0.02, range: 0.001-0.15")
    print("- Cube base: 0.15, range: 0.05-0.4") 
    print("- Intensity variation: ±0.2")
    print()
    
    test_new_intensities()

if __name__ == "__main__":
    main() 