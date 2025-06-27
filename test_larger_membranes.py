#!/usr/bin/env python3
"""Test script to compare membrane sizes before and after parameter changes"""

import sys
sys.path.insert(0, 'venv/lib/python3.10/site-packages')
import numpy as np
from vol_generator import MembraneGen

def test_membrane_sizes():
    print("Testing membrane size changes...")
    
    # Current generator (with larger membranes)
    gen_new = MembraneGen(generate_masks=True)
    
    # Create old generator for comparison
    gen_old = MembraneGen(generate_masks=True)
    gen_old.sigma = (20, 25)  # Old sigma values
    gen_old.band = 0.1        # Old band value
    
    seed = 42
    
    # Generate with new parameters
    print("\nGenerating with NEW parameters (larger membranes):")
    vol_new_bytes, mask_new_bytes = gen_new(seed)
    vol_new = np.frombuffer(vol_new_bytes, dtype=np.float32).reshape(96, 96, 96)
    mask_new = np.frombuffer(mask_new_bytes, dtype=np.uint8).reshape(96, 96, 96)
    
    counts_new = np.bincount(mask_new.flatten())
    total_voxels = mask_new.size
    
    print(f"  Volume range: [{vol_new.min():.3f}, {vol_new.max():.3f}]")
    print(f"  Mask labels: {np.unique(mask_new)}")
    print(f"  Background: {counts_new[0]:,} ({counts_new[0]/total_voxels*100:.1f}%)")
    print(f"  Membrane:   {counts_new[1]:,} ({counts_new[1]/total_voxels*100:.1f}%)")
    print(f"  Spheres:    {counts_new[2]:,} ({counts_new[2]/total_voxels*100:.1f}%)")
    print(f"  Cubes:      {counts_new[3]:,} ({counts_new[3]/total_voxels*100:.1f}%)")
    
    # Generate with old parameters
    print("\nGenerating with OLD parameters (smaller membranes):")
    vol_old_bytes, mask_old_bytes = gen_old(seed)
    vol_old = np.frombuffer(vol_old_bytes, dtype=np.float32).reshape(96, 96, 96)
    mask_old = np.frombuffer(mask_old_bytes, dtype=np.uint8).reshape(96, 96, 96)
    
    counts_old = np.bincount(mask_old.flatten())
    
    print(f"  Volume range: [{vol_old.min():.3f}, {vol_old.max():.3f}]")
    print(f"  Mask labels: {np.unique(mask_old)}")
    print(f"  Background: {counts_old[0]:,} ({counts_old[0]/total_voxels*100:.1f}%)")
    print(f"  Membrane:   {counts_old[1]:,} ({counts_old[1]/total_voxels*100:.1f}%)")
    print(f"  Spheres:    {counts_old[2]:,} ({counts_old[2]/total_voxels*100:.1f}%)")
    print(f"  Cubes:      {counts_old[3]:,} ({counts_old[3]/total_voxels*100:.1f}%)")
    
    # Compare
    print(f"\n=== COMPARISON ===")
    membrane_increase = counts_new[1] / counts_old[1]
    print(f"Membrane voxel count increased by {membrane_increase:.1f}x")
    print(f"Membrane percentage: {counts_old[1]/total_voxels*100:.1f}% → {counts_new[1]/total_voxels*100:.1f}%")
    
    # Check if we got a good increase
    if counts_new[1] > counts_old[1] * 1.5:
        print("✅ SUCCESS: Membranes are significantly larger!")
    elif counts_new[1] > counts_old[1]:
        print("⚠️  PARTIAL: Membranes are larger but maybe increase sigma more")
    else:
        print("❌ ISSUE: No membrane size increase detected")

if __name__ == "__main__":
    test_membrane_sizes() 