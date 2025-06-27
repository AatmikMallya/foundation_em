#!/usr/bin/env python3
"""Simple test for cube generation without matplotlib"""

import sys
import os
sys.path.insert(0, 'venv/lib/python3.10/site-packages')

import numpy as np
from vol_generator import MembraneGen

# Test the updated generator
print("Testing cube generation...")
gen = MembraneGen(generate_masks=True)

for seed in [42, 123, 456]:
    print(f"\nSeed {seed}:")
    vol_bytes, mask_bytes = gen(seed)
    
    volume = np.frombuffer(vol_bytes, dtype=np.float32).reshape(96, 96, 96)
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
    
    print(f"  Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
    print(f"  Mask labels: {np.unique(mask)}")
    print(f"  Label counts: {np.bincount(mask.flatten())}")
    
    # Check if we have all expected shapes
    unique_labels = set(np.unique(mask))
    expected_labels = {0, 1, 2, 3}  # bg, membrane, sphere, cube
    
    if expected_labels.issubset(unique_labels):
        print(f"  ✓ All shape types present")
    else:
        missing = expected_labels - unique_labels
        print(f"  ⚠ Missing labels: {missing}")

print("\nCube generation test complete!") 