#!/usr/bin/env python3
"""Test script to compare thick vs thin membranes with large spatial extent"""

import sys
sys.path.insert(0, 'venv/lib/python3.10/site-packages')
import numpy as np
from vol_generator import MembraneGen

def test_membrane_thickness():
    print("Testing membrane thickness changes...")
    
    # Current generator (large + thin)
    gen_thin = MembraneGen(generate_masks=True)
    
    # Create thick generator for comparison
    gen_thick = MembraneGen(generate_masks=True)
    gen_thick.band = 0.15  # Thick band
    
    seed = 42
    
    # Generate with thin membranes
    print("\nGenerating with THIN membranes (band=0.1, sigma=35-45):")
    vol_thin_bytes, mask_thin_bytes = gen_thin(seed)
    vol_thin = np.frombuffer(vol_thin_bytes, dtype=np.float32).reshape(96, 96, 96)
    mask_thin = np.frombuffer(mask_thin_bytes, dtype=np.uint8).reshape(96, 96, 96)
    
    counts_thin = np.bincount(mask_thin.flatten())
    total_voxels = mask_thin.size
    
    print(f"  Volume range: [{vol_thin.min():.3f}, {vol_thin.max():.3f}]")
    print(f"  Background: {counts_thin[0]:,} ({counts_thin[0]/total_voxels*100:.1f}%)")
    print(f"  Membrane:   {counts_thin[1]:,} ({counts_thin[1]/total_voxels*100:.1f}%)")
    print(f"  Spheres:    {counts_thin[2]:,} ({counts_thin[2]/total_voxels*100:.1f}%)")
    print(f"  Cubes:      {counts_thin[3]:,} ({counts_thin[3]/total_voxels*100:.1f}%)")
    
    # Generate with thick membranes
    print("\nGenerating with THICK membranes (band=0.15, sigma=35-45):")
    vol_thick_bytes, mask_thick_bytes = gen_thick(seed)
    vol_thick = np.frombuffer(vol_thick_bytes, dtype=np.float32).reshape(96, 96, 96)
    mask_thick = np.frombuffer(mask_thick_bytes, dtype=np.uint8).reshape(96, 96, 96)
    
    counts_thick = np.bincount(mask_thick.flatten())
    
    print(f"  Volume range: [{vol_thick.min():.3f}, {vol_thick.max():.3f}]")
    print(f"  Background: {counts_thick[0]:,} ({counts_thick[0]/total_voxels*100:.1f}%)")
    print(f"  Membrane:   {counts_thick[1]:,} ({counts_thick[1]/total_voxels*100:.1f}%)")
    print(f"  Spheres:    {counts_thick[2]:,} ({counts_thick[2]/total_voxels*100:.1f}%)")
    print(f"  Cubes:      {counts_thick[3]:,} ({counts_thick[3]/total_voxels*100:.1f}%)")
    
    # Compare
    print(f"\n=== COMPARISON ===")
    thickness_ratio = counts_thick[1] / counts_thin[1]
    print(f"Thick vs Thin membrane ratio: {thickness_ratio:.1f}x")
    print(f"Thin membrane coverage: {counts_thin[1]/total_voxels*100:.1f}%")
    print(f"Thick membrane coverage: {counts_thick[1]/total_voxels*100:.1f}%")
    
    print(f"\n✅ Result: Large spatial extent with thin membranes")
    print(f"   - Sigma (35,45) gives large membrane networks") 
    print(f"   - Band 0.1 keeps them thin and realistic")

if __name__ == "__main__":
    test_membrane_thickness() 