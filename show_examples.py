#!/usr/bin/env python3
"""
Show examples of all structure combinations in synthetic EM data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from vol_generator import MembraneGen

def generate_combination_examples():
    """Generate examples of each structure combination"""
    print("Generating examples of each combination...")
    gen = MembraneGen(generate_masks=True, equal_combinations=False)  # We'll control combinations manually
    
    # Define all 8 combinations with descriptive names
    combinations = [
        {'name': 'Background Only', 'membrane': False, 'spheres': False, 'cubes': False},
        {'name': 'Membranes Only', 'membrane': True, 'spheres': False, 'cubes': False},
        {'name': 'Spheres Only', 'membrane': False, 'spheres': True, 'cubes': False},
        {'name': 'Cubes Only', 'membrane': False, 'spheres': False, 'cubes': True},
        {'name': 'Membranes + Spheres', 'membrane': True, 'spheres': True, 'cubes': False},
        {'name': 'Membranes + Cubes', 'membrane': True, 'spheres': False, 'cubes': True},
        {'name': 'Spheres + Cubes', 'membrane': False, 'spheres': True, 'cubes': True},
        {'name': 'All Structures', 'membrane': True, 'spheres': True, 'cubes': True},
    ]
    
    volumes = []
    masks = []
    combo_info = []
    
    # Generate one example of each combination
    for i, combo in enumerate(combinations):
        print(f"  Generating {combo['name']} (seed {i*100 + 42})")
        
        # Temporarily modify the generator to force this combination
        old_combinations = gen.combinations
        gen.combinations = [combo]  # Force this specific combination
        
        seed = i * 100 + 42
        vol_bytes, mask_bytes = gen(seed)
        
        volume = np.frombuffer(vol_bytes, dtype=np.float32).reshape(96, 96, 96)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
        
        volumes.append(volume)
        masks.append(mask)
        combo_info.append(combo)
        
        # Print stats
        unique_labels = np.unique(mask)
        counts = np.bincount(mask.flatten())
        print(f"    Labels: {unique_labels}, Counts: {counts[:len(unique_labels)]}")
        
        # Restore original combinations
        gen.combinations = old_combinations
    
    return volumes, masks, combo_info

def create_static_examples():
    """Create static slice examples showing different combinations"""
    print("Creating static slice examples...")
    
    volumes, masks, combo_info = generate_combination_examples()
    
    # Create a large figure showing middle slices of each combination
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Create colormap for masks
    colors = ['black', 'blue', 'red', 'orange']  # bg, membrane, sphere, cube
    cmap = mcolors.ListedColormap(colors)
    
    for i, (volume, mask, combo) in enumerate(zip(volumes, masks, combo_info)):
        ax = axes[i]
        
        # Show middle slice (z=48)
        z_slice = 48
        vol_norm = (volume - volume.min()) / (volume.max() - volume.min())
        
        # Show overlay
        ax.imshow(vol_norm[z_slice], cmap='gray', vmin=0, vmax=1, alpha=0.8)
        
        # Overlay shapes with transparency
        membrane_mask = mask[z_slice] == 1
        sphere_mask = mask[z_slice] == 2
        cube_mask = mask[z_slice] == 3
        
        if membrane_mask.any():
            ax.imshow(np.where(membrane_mask, 1, np.nan), cmap='Blues', alpha=0.6, vmin=0, vmax=1)
        if sphere_mask.any():
            ax.imshow(np.where(sphere_mask, 1, np.nan), cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        if cube_mask.any():
            ax.imshow(np.where(cube_mask, 1, np.nan), cmap='Oranges', alpha=0.6, vmin=0, vmax=1)
        
        ax.set_title(f"{combo['name']}\n(Middle slice Z={z_slice})", fontsize=12)
        ax.axis('off')
        
        # Add stats text
        unique_labels = np.unique(mask)
        counts = np.bincount(mask.flatten())
        stats_text = []
        label_names = ['BG', 'Mem', 'Sph', 'Cube']
        for j, count in enumerate(counts[:len(unique_labels)]):
            if j < len(label_names):
                pct = (count / mask.size) * 100
                stats_text.append(f"{label_names[j]}: {pct:.1f}%")
        
        ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
    
    plt.suptitle('Synthetic EM Volume Examples: All Structure Combinations\n(Blue=Membrane, Red=Sphere, Orange=Cube)', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('structure_combinations_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Static examples saved as structure_combinations_examples.png")
    return 'structure_combinations_examples.png'

def show_random_samples():
    """Show some random samples using equal combinations"""
    print("\nGenerating random samples with equal combinations...")
    gen = MembraneGen(generate_masks=True, equal_combinations=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    seeds = [42, 123, 456, 789, 1000, 1337]
    
    for i, seed in enumerate(seeds):
        print(f"  Sample {i+1} (seed {seed})")
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
        
        ax.set_title(f"Sample {i+1}\n{' + '.join(structures)}", fontsize=12)
        ax.axis('off')
        
        # Print stats
        counts = np.bincount(mask.flatten())
        print(f"    Labels: {unique_labels}, Counts: {counts[:len(unique_labels)]}")
    
    plt.suptitle('Random Samples from Equal Combinations Generator\n(Blue=Membrane, Red=Sphere, Orange=Cube)', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('random_samples_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Random samples saved as random_samples_examples.png")
    return 'random_samples_examples.png'

def main():
    """Main function to create example visualizations"""
    print("=== Synthetic EM Data Examples ===")
    
    # Create examples of all combinations
    create_static_examples()
    
    # Show random samples
    show_random_samples()
    
    print("\n✓ Complete! Generated two visualization files:")
    print("  - structure_combinations_examples.png: All 8 possible combinations")
    print("  - random_samples_examples.png: Random samples showing variety")

if __name__ == "__main__":
    main() 