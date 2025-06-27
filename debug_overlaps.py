#!/usr/bin/env python3
"""Debug script to check for overlaps in generated volumes"""

import sys
sys.path.insert(0, 'venv/lib/python3.10/site-packages')
import numpy as np
from vol_generator import MembraneGen

def check_all_overlaps():
    print("Debugging overlaps in generated volumes...")
    
    gen = MembraneGen(generate_masks=True)
    
    for seed in [42, 123, 456]:
        print(f"\n=== Checking Volume (seed {seed}) ===")
        
        vol_bytes, mask_bytes = gen(seed)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
        
        # Check for all possible overlaps
        bg_mask = mask == 0     # background
        mem_mask = mask == 1    # membrane  
        sph_mask = mask == 2    # sphere
        cube_mask = mask == 3   # cube
        
        # Count each type
        counts = np.bincount(mask.flatten())
        total = mask.size
        
        print(f"  Background: {counts[0]:,} ({counts[0]/total*100:.1f}%)")
        print(f"  Membrane:   {counts[1]:,} ({counts[1]/total*100:.1f}%)")
        print(f"  Spheres:    {counts[2]:,} ({counts[2]/total*100:.1f}%)")
        print(f"  Cubes:      {counts[3]:,} ({counts[3]/total*100:.1f}%)")
        
        # Check for impossible overlaps (each voxel should have exactly one label)
        total_labeled = counts[0] + counts[1] + counts[2] + counts[3]
        print(f"  Total labeled voxels: {total_labeled:,}")
        print(f"  Expected total: {total:,}")
        
        if total_labeled != total:
            print(f"  ❌ ISSUE: Missing {total - total_labeled:,} voxels!")
        
        # Check for any voxels with multiple labels (shouldn't be possible)
        # This would indicate a bug in our mask assignment
        unique_labels = np.unique(mask)
        print(f"  Unique labels found: {unique_labels}")
        
        if len(unique_labels) > 4:
            print(f"  ❌ ISSUE: Found unexpected labels: {unique_labels}")
        
        # Visual check: look for patterns that suggest overlaps
        # Check some middle slices
        for z in [24, 48, 72]:
            slice_mask = mask[z]
            slice_unique = np.unique(slice_mask)
            
            # Count transitions between different labels (should be clean boundaries)
            transitions = 0
            for i in range(slice_mask.shape[0]-1):
                for j in range(slice_mask.shape[1]-1):
                    current = slice_mask[i, j]
                    right = slice_mask[i, j+1]
                    down = slice_mask[i+1, j]
                    
                    # Count how many different labels are adjacent
                    neighbors = {current, right, down}
                    if len(neighbors) > 2:  # More than 2 different labels touching
                        transitions += 1
            
            print(f"  Slice {z}: {len(slice_unique)} labels, {transitions} complex transitions")
        
        # Check if spheres and cubes are actually separated
        sphere_cube_overlap = np.sum(sph_mask & cube_mask)
        mem_sphere_overlap = np.sum(mem_mask & sph_mask)
        mem_cube_overlap = np.sum(mem_mask & cube_mask)
        
        print(f"  Sphere-Cube overlaps: {sphere_cube_overlap}")
        print(f"  Membrane-Sphere overlaps: {mem_sphere_overlap}")
        print(f"  Membrane-Cube overlaps: {mem_cube_overlap}")
        
        if sphere_cube_overlap > 0:
            print(f"  ❌ SPHERE-CUBE OVERLAP DETECTED!")
        if mem_sphere_overlap > 0:
            print(f"  ⚠️  Membrane-Sphere overlap (expected due to our current approach)")
        if mem_cube_overlap > 0:
            print(f"  ⚠️  Membrane-Cube overlap (expected due to our current approach)")

if __name__ == "__main__":
    check_all_overlaps() 