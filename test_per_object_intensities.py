#!/usr/bin/env python3
"""
Test that individual objects have different intensities within the same volume
"""

import numpy as np
import matplotlib.pyplot as plt
from vol_generator import MembraneGen
from scipy.ndimage import label

def test_per_object_intensities():
    """Test that each object has its own intensity"""
    print("Testing per-object intensity variation...")
    
    gen = MembraneGen(generate_masks=True, equal_combinations=False)
    
    # Generate a single volume with both spheres and cubes
    vol_bytes, mask_bytes = gen(42)  # Fixed seed for reproducible test
    volume = np.frombuffer(vol_bytes, dtype=np.float32).reshape(96, 96, 96)
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
    
    print(f"Volume contains: {np.unique(mask)}")
    
    # Analyze spheres
    if 2 in mask:
        sphere_mask = mask == 2
        # Use connected components to identify individual spheres
        labeled_spheres, num_spheres = label(sphere_mask)
        
        sphere_intensities = []
        for sphere_id in range(1, num_spheres + 1):
            sphere_voxels = volume[labeled_spheres == sphere_id]
            if len(sphere_voxels) > 0:
                mean_intensity = np.mean(sphere_voxels)
                sphere_intensities.append(mean_intensity)
        
        print(f"\nSpheres ({num_spheres} found):")
        for i, intensity in enumerate(sphere_intensities):
            print(f"  Sphere {i+1}: mean intensity = {intensity:.4f}")
        
        if len(sphere_intensities) > 1:
            sphere_range = max(sphere_intensities) - min(sphere_intensities)
            print(f"  Range: {sphere_range:.4f}")
            print(f"  Std: {np.std(sphere_intensities):.4f}")
    
    # Analyze cubes
    if 3 in mask:
        cube_mask = mask == 3
        # Use connected components to identify individual cubes
        labeled_cubes, num_cubes = label(cube_mask)
        
        cube_intensities = []
        for cube_id in range(1, num_cubes + 1):
            cube_voxels = volume[labeled_cubes == cube_id]
            if len(cube_voxels) > 0:
                mean_intensity = np.mean(cube_voxels)
                cube_intensities.append(mean_intensity)
        
        print(f"\nCubes ({num_cubes} found):")
        for i, intensity in enumerate(cube_intensities):
            print(f"  Cube {i+1}: mean intensity = {intensity:.4f}")
        
        if len(cube_intensities) > 1:
            cube_range = max(cube_intensities) - min(cube_intensities)
            print(f"  Range: {cube_range:.4f}")
            print(f"  Std: {np.std(cube_intensities):.4f}")
    
    # Visualize middle slice
    z_slice = 48
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Volume
    ax = axes[0]
    im = ax.imshow(volume[z_slice], cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Volume (Raw Intensities)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    # Mask
    ax = axes[1]
    ax.imshow(mask[z_slice], cmap='jet', vmin=0, vmax=3)
    ax.set_title('Mask')
    ax.axis('off')
    
    # Overlay
    ax = axes[2]
    ax.imshow(volume[z_slice], cmap='gray', alpha=0.8)
    sphere_slice = mask[z_slice] == 2
    cube_slice = mask[z_slice] == 3
    if sphere_slice.any():
        ax.imshow(np.where(sphere_slice, 1, np.nan), cmap='Reds', alpha=0.6)
    if cube_slice.any():
        ax.imshow(np.where(cube_slice, 1, np.nan), cmap='Oranges', alpha=0.6)
    ax.set_title('Overlay')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('per_object_intensity_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Per-object intensity test saved as per_object_intensity_test.png")

def test_multiple_volumes():
    """Test intensity variation across multiple volumes"""
    print("\nTesting intensity variation across multiple volumes...")
    
    gen = MembraneGen(generate_masks=True, equal_combinations=False)
    
    all_sphere_means = []
    all_cube_means = []
    
    for seed in range(10):
        vol_bytes, mask_bytes = gen(seed + 100)
        volume = np.frombuffer(vol_bytes, dtype=np.float32).reshape(96, 96, 96)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
        
        # Get mean intensities per object type per volume
        if 2 in mask:
            sphere_voxels = volume[mask == 2]
            sphere_mean = np.mean(sphere_voxels)
            all_sphere_means.append(sphere_mean)
        
        if 3 in mask:
            cube_voxels = volume[mask == 3]
            cube_mean = np.mean(cube_voxels)
            all_cube_means.append(cube_mean)
    
    print(f"Sphere means across volumes: {[f'{x:.3f}' for x in all_sphere_means]}")
    print(f"Cube means across volumes: {[f'{x:.3f}' for x in all_cube_means]}")
    
    if all_sphere_means:
        print(f"Sphere variation: {np.std(all_sphere_means):.4f}")
    if all_cube_means:
        print(f"Cube variation: {np.std(all_cube_means):.4f}")

def main():
    print("=== Testing Per-Object Intensity Variation ===")
    print("Expected behavior:")
    print("- Each sphere should have different intensity (base ± 0.1)")
    print("- Each cube should have different intensity (base ± 0.14)")
    print("- Objects should vary both within and between volumes")
    print()
    
    test_per_object_intensities()
    test_multiple_volumes()

if __name__ == "__main__":
    main() 