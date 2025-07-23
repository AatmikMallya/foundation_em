#!/usr/bin/env python3
"""
Create summary examples showing the updated synthetic EM data
"""

import numpy as np
import matplotlib.pyplot as plt
from vol_generator import MembraneGen

def show_parameter_changes():
    """Show the key parameter changes made to the generator"""
    print("=== Updated Synthetic EM Data Parameters ===")
    print()
    print("Sphere parameters:")
    print("  - Count: 2-16 spheres (was 4-8)")
    print("  - Radius: 4-12 pixels")
    print()
    print("Cube parameters:")
    print("  - Count: 2-16 cubes (was 2-6)")
    print("  - Size: 8-16 pixels")
    print()
    print("Equal combinations mode: ENABLED")
    print("  - All 8 structure combinations generated with equal probability:")
    print("    1. Background only")
    print("    2. Membranes only")
    print("    3. Spheres only")
    print("    4. Cubes only")
    print("    5. Membranes + Spheres")
    print("    6. Membranes + Cubes")
    print("    7. Spheres + Cubes")
    print("    8. All structures")
    print()

def create_sample_grid():
    """Create a grid of sample volumes"""
    print("Generating sample grid...")
    
    gen = MembraneGen(generate_masks=True, equal_combinations=True)
    
    # Generate 9 random samples
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    seeds = [42, 123, 456, 789, 1000, 1337, 2000, 2500, 3000]
    
    for i, seed in enumerate(seeds):
        vol_bytes, mask_bytes = gen(seed)
        volume = np.frombuffer(vol_bytes, dtype=np.float32).reshape(96, 96, 96)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
        
        # Show middle slice
        z_slice = 48
        vol_norm = (volume - volume.min()) / (volume.max() - volume.min())
        
        ax = axes[i]
        ax.imshow(vol_norm[z_slice], cmap='gray', vmin=0, vmax=1, alpha=0.8)
        
        # Overlay shapes
        membrane_mask = mask[z_slice] == 1
        sphere_mask = mask[z_slice] == 2
        cube_mask = mask[z_slice] == 3
        
        if membrane_mask.any():
            ax.imshow(np.where(membrane_mask, 1, np.nan), cmap='Blues', alpha=0.6, vmin=0, vmax=1)
        if sphere_mask.any():
            ax.imshow(np.where(sphere_mask, 1, np.nan), cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        if cube_mask.any():
            ax.imshow(np.where(cube_mask, 1, np.nan), cmap='Oranges', alpha=0.6, vmin=0, vmax=1)
        
        # Determine which structures are present
        unique_labels = np.unique(mask)
        structures = []
        if 1 in unique_labels: structures.append('Mem')
        if 2 in unique_labels: structures.append('Sph')
        if 3 in unique_labels: structures.append('Cube')
        if not structures: structures = ['BG Only']
        
        ax.set_title(f"Seed {seed}\n{' + '.join(structures)}", fontsize=11)
        ax.axis('off')
        
        # Print stats
        counts = np.bincount(mask.flatten())
        total_organelles = sum(counts[1:]) if len(counts) > 1 else 0
        bg_pct = (counts[0] / mask.size) * 100
        organelle_pct = (total_organelles / mask.size) * 100
        print(f"  Seed {seed}: {bg_pct:.1f}% bg, {organelle_pct:.1f}% structures ({', '.join(structures)})")
    
    plt.suptitle('Updated Synthetic EM Data: Random Samples\n(Blue=Membrane, Red=Sphere, Orange=Cube)', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('sample_grid_updated.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Sample grid saved as sample_grid_updated.png")

def analyze_volume_statistics():
    """Analyze statistics across multiple volumes"""
    print("\nAnalyzing volume statistics...")
    
    gen = MembraneGen(generate_masks=True, equal_combinations=True)
    
    # Generate 100 volumes to get statistics
    combination_counts = {}
    total_volumes = 100
    
    for seed in range(total_volumes):
        vol_bytes, mask_bytes = gen(seed)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
        
        # Determine combination
        unique_labels = set(np.unique(mask))
        has_mem = 1 in unique_labels
        has_sph = 2 in unique_labels
        has_cube = 3 in unique_labels
        
        combo_key = (has_mem, has_sph, has_cube)
        combination_counts[combo_key] = combination_counts.get(combo_key, 0) + 1
    
    print(f"\nCombination frequencies (out of {total_volumes} volumes):")
    combo_names = {
        (False, False, False): "Background Only",
        (True, False, False): "Membranes Only",
        (False, True, False): "Spheres Only", 
        (False, False, True): "Cubes Only",
        (True, True, False): "Membranes + Spheres",
        (True, False, True): "Membranes + Cubes",
        (False, True, True): "Spheres + Cubes",
        (True, True, True): "All Structures"
    }
    
    for combo, count in sorted(combination_counts.items()):
        name = combo_names.get(combo, f"Unknown {combo}")
        pct = (count / total_volumes) * 100
        print(f"  {name}: {count} volumes ({pct:.1f}%)")

def main():
    """Main function"""
    show_parameter_changes()
    create_sample_grid()
    analyze_volume_statistics()
    
    print("\n✓ Summary complete!")
    print("Files generated:")
    print("  - sample_grid_updated.png: 3x3 grid of random samples")
    print("  - Terminal output: Parameter changes and statistics")

if __name__ == "__main__":
    main() 