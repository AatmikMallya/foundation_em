#!/usr/bin/env python3

import torch
import numpy as np
from membrane_synthetic_data import create_membrane_dataloader

def test_on_the_fly_generation():
    """Test that data is actually generated on-the-fly and changes between epochs."""
    
    # Create a small dataset for testing
    loader = create_membrane_dataloader(
        batch_size=2,
        num_samples=4,
        volume_size=(32, 32, 32),
        num_gaussians_range=(3, 5),
        gaussian_sigma_range=(3, 8),
        noise_level=0.01,
        membrane_band_width=0.1,
        num_workers=0,
        shuffle=False,  # No shuffle for deterministic testing
        seed=42,
        num_additional_spheres_range=(1, 3),
        additional_sphere_radius_range=(2.0, 4.0)
    )
    
    print("Testing on-the-fly data generation...")
    print("=" * 50)
    
    # Test that data changes between epochs
    epoch_data_sums = []
    
    for epoch in range(3):
        print(f"Epoch {epoch}:")
        loader.dataset.set_epoch(epoch)
        
        batch_sums = []
        for batch_idx, batch in enumerate(loader):
            batch_sum = batch.sum().item()
            batch_sums.append(batch_sum)
            print(f"  Batch {batch_idx}: sum = {batch_sum:.4f}")
        
        epoch_sum = sum(batch_sums)
        epoch_data_sums.append(epoch_sum)
        print(f"  Total epoch sum: {epoch_sum:.4f}")
        print()
    
    # Check if data is actually different between epochs
    print("Analysis:")
    all_same = all(abs(s - epoch_data_sums[0]) < 1e-6 for s in epoch_data_sums)
    
    if all_same:
        print("❌ ERROR: Data is IDENTICAL across epochs!")
        print("   This means on-the-fly generation is NOT working.")
    else:
        print("✅ SUCCESS: Data is DIFFERENT across epochs!")
        print("   On-the-fly generation is working correctly.")
    
    print(f"Epoch sums: {epoch_data_sums}")
    
    # Test that data is deterministic for the same epoch
    print("\nTesting determinism within same epoch...")
    loader.dataset.set_epoch(0)
    first_batch = next(iter(loader))
    
    loader.dataset.set_epoch(0)  # Reset to same epoch
    second_batch = next(iter(loader))
    
    if torch.allclose(first_batch, second_batch):
        print("✅ SUCCESS: Data is deterministic within same epoch!")
    else:
        print("❌ ERROR: Data is NOT deterministic within same epoch!")
    
    return not all_same

if __name__ == '__main__':
    test_on_the_fly_generation() 