#!/usr/bin/env python3
"""
Test script to verify mask generation works correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from vol_generator import MembraneGen

def test_mask_generation():
    """Test that mask generation produces reasonable results"""
    
    # Create generator with mask generation enabled
    gen = MembraneGen(generate_masks=True)
    
    # Generate a few test volumes with masks
    seeds = [42, 123, 456]
    
    fig, axes = plt.subplots(len(seeds), 6, figsize=(20, 4*len(seeds)))
    if len(seeds) == 1:
        axes = axes.reshape(1, -1)
    
    for i, seed in enumerate(seeds):
        print(f"Generating volume with seed {seed}...")
        
        vol_bytes, mask_bytes = gen(seed)
        
        # Convert back to arrays
        volume = np.frombuffer(vol_bytes, dtype=np.float32).reshape(96, 96, 96)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
        
        print(f"Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
        print(f"Mask labels: {np.unique(mask)}")
        print(f"Label counts: {np.bincount(mask.flatten())}")
        
        # Show middle slice
        mid_slice = 48
        
        # Volume
        axes[i, 0].imshow(volume[mid_slice], cmap='gray')
        axes[i, 0].set_title(f'Volume (seed {seed})')
        axes[i, 0].axis('off')
        
        # Mask
        axes[i, 1].imshow(mask[mid_slice], cmap='tab10', vmin=0, vmax=3)
        axes[i, 1].set_title(f'Mask (0=bg, 1=mem, 2=sph, 3=cube)')
        axes[i, 1].axis('off')
        
        # Membrane overlay
        membrane_mask = mask[mid_slice] == 1
        overlay = volume[mid_slice].copy()
        overlay[membrane_mask] = 1.0  # Highlight membranes
        axes[i, 2].imshow(overlay, cmap='gray')
        axes[i, 2].set_title('Membrane highlighted')
        axes[i, 2].axis('off')
        
        # Sphere overlay
        sphere_mask = mask[mid_slice] == 2
        overlay = volume[mid_slice].copy()
        overlay[sphere_mask] = 1.0  # Highlight spheres
        axes[i, 3].imshow(overlay, cmap='gray')
        axes[i, 3].set_title('Spheres highlighted')
        axes[i, 3].axis('off')
        
        # Cube overlay
        cube_mask = mask[mid_slice] == 3
        overlay = volume[mid_slice].copy()
        overlay[cube_mask] = 1.0  # Highlight cubes
        axes[i, 4].imshow(overlay, cmap='gray')
        axes[i, 4].set_title('Cubes highlighted')
        axes[i, 4].axis('off')
        
        # All shapes combined
        combined_overlay = volume[mid_slice].copy()
        combined_overlay[membrane_mask] = 0.8
        combined_overlay[sphere_mask] = 0.9
        combined_overlay[cube_mask] = 1.0
        axes[i, 5].imshow(combined_overlay, cmap='gray')
        axes[i, 5].set_title('All shapes')
        axes[i, 5].axis('off')
        
        print()
    
    plt.tight_layout()
    plt.savefig('test_mask_generation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Mask generation test complete!")
    print("Check test_mask_generation.png for visual verification")

if __name__ == "__main__":
    test_mask_generation() 