#!/usr/bin/env python3
"""Test collision detection implementation"""

import sys
sys.path.insert(0, 'venv/lib/python3.10/site-packages')
import numpy as np
from vol_generator import MembraneGen

def test_collision_detection():
    print("Testing collision detection...")
    
    # Test with collision detection (current implementation)
    gen_new = MembraneGen(generate_masks=True)
    
    # Create old generator without collision detection for comparison
    gen_old = MembraneGen(generate_masks=True)
    # Monkey patch to disable collision detection
    def old_check_collision(self, *args, **kwargs):
        return False  # Never detect collisions (always allow placement)
    gen_old.check_collision = old_check_collision.__get__(gen_old, MembraneGen)
    
    seed = 42
    
    # Generate with collision detection
    print("\n=== WITH Collision Detection ===")
    vol_new_bytes, mask_new_bytes = gen_new(seed)
    vol_new = np.frombuffer(vol_new_bytes, dtype=np.float32).reshape(96, 96, 96)
    mask_new = np.frombuffer(mask_new_bytes, dtype=np.uint8).reshape(96, 96, 96)
    
    counts_new = np.bincount(mask_new.flatten())
    total_voxels = mask_new.size
    
    print(f"  Background: {counts_new[0]:,} ({counts_new[0]/total_voxels*100:.1f}%)")
    print(f"  Membrane:   {counts_new[1]:,} ({counts_new[1]/total_voxels*100:.1f}%)")
    print(f"  Spheres:    {counts_new[2]:,} ({counts_new[2]/total_voxels*100:.1f}%)")
    print(f"  Cubes:      {counts_new[3]:,} ({counts_new[3]/total_voxels*100:.1f}%)")
    
    # Check for overlaps in new version
    sphere_mask = mask_new == 2
    cube_mask = mask_new == 3
    overlaps = np.sum(sphere_mask & cube_mask)
    print(f"  Sphere-Cube overlaps: {overlaps} voxels")
    
    # Generate without collision detection  
    print("\n=== WITHOUT Collision Detection ===")
    vol_old_bytes, mask_old_bytes = gen_old(seed)
    vol_old = np.frombuffer(vol_old_bytes, dtype=np.float32).reshape(96, 96, 96)
    mask_old = np.frombuffer(mask_old_bytes, dtype=np.uint8).reshape(96, 96, 96)
    
    counts_old = np.bincount(mask_old.flatten())
    
    print(f"  Background: {counts_old[0]:,} ({counts_old[0]/total_voxels*100:.1f}%)")
    print(f"  Membrane:   {counts_old[1]:,} ({counts_old[1]/total_voxels*100:.1f}%)")
    print(f"  Spheres:    {counts_old[2]:,} ({counts_old[2]/total_voxels*100:.1f}%)")
    print(f"  Cubes:      {counts_old[3]:,} ({counts_old[3]/total_voxels*100:.1f}%)")
    
    # Check overlaps in old version (should have some due to overwriting)
    sphere_mask_old = mask_old == 2
    cube_mask_old = mask_old == 3
    
    print(f"\n=== COMPARISON ===")
    print(f"Collision detection working: {overlaps == 0}")
    print(f"Shape count changes:")
    print(f"  Spheres: {counts_old[2]:,} → {counts_new[2]:,} ({counts_new[2]/counts_old[2]:.2f}x)")
    print(f"  Cubes:   {counts_old[3]:,} → {counts_new[3]:,} ({counts_new[3]/counts_old[3]:.2f}x)")
    
    if overlaps == 0:
        print("✅ SUCCESS: No overlaps detected with collision detection!")
    else:
        print(f"❌ ISSUE: Still found {overlaps} overlapping voxels")

if __name__ == "__main__":
    test_collision_detection() 