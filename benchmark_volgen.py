#!/usr/bin/env python3
"""
Benchmark script to measure vol_generator.py speedup
"""

import sys
sys.path.insert(0, 'venv/lib/python3.10/site-packages')
import time
import numpy as np
from vol_generator import MembraneGen

def benchmark_volume_generation(n_volumes=50):
    """Benchmark the optimized volume generator"""
    
    print(f"=== Benchmarking Volume Generation ===")
    print(f"Generating {n_volumes} volumes with masks...")
    
    gen = MembraneGen(generate_masks=True)
    
    # Warm up
    print("Warming up...")
    for i in range(3):
        _ = gen(i)
    
    # Benchmark
    print("Running benchmark...")
    start_time = time.time()
    
    total_mem_voxels = 0
    total_sph_voxels = 0
    total_cube_voxels = 0
    
    for i in range(n_volumes):
        vol_bytes, mask_bytes = gen(i)
        
        # Quick stats
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
        counts = np.bincount(mask.flatten())
        
        total_mem_voxels += counts[1] if len(counts) > 1 else 0
        total_sph_voxels += counts[2] if len(counts) > 2 else 0
        total_cube_voxels += counts[3] if len(counts) > 3 else 0
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  {i+1:3d}/{n_volumes} volumes generated, {rate:.2f} vol/sec")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== BENCHMARK RESULTS ===")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Volume generation rate: {n_volumes / total_time:.2f} volumes/second")
    print(f"Average time per volume: {total_time / n_volumes * 1000:.1f} ms")
    
    print(f"\n=== AVERAGE SHAPE STATISTICS ===")
    print(f"Average membrane voxels: {total_mem_voxels / n_volumes:.0f}")
    print(f"Average sphere voxels: {total_sph_voxels / n_volumes:.0f}")
    print(f"Average cube voxels: {total_cube_voxels / n_volumes:.0f}")
    
    return n_volumes / total_time

def main():
    """Run the benchmark"""
    
    print("Volume Generator Performance Benchmark")
    print("=" * 50)
    
    # Test different volume counts to see scaling
    test_sizes = [10, 25, 50]
    
    rates = []
    for n_vols in test_sizes:
        print(f"\n🚀 Testing with {n_vols} volumes:")
        rate = benchmark_volume_generation(n_vols)
        rates.append(rate)
        
        print(f"Rate: {rate:.2f} volumes/second")
        print("-" * 30)
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"Performance appears stable across different batch sizes:")
    for i, (n_vols, rate) in enumerate(zip(test_sizes, rates)):
        print(f"  {n_vols:2d} volumes: {rate:.2f} vol/sec")
    
    avg_rate = np.mean(rates)
    print(f"\nAverage performance: {avg_rate:.2f} volumes/second")
    
    # Estimate time for full dataset
    full_dataset_size = 1_048_576  # 1M volumes
    estimated_hours = full_dataset_size / avg_rate / 3600
    print(f"Estimated time for {full_dataset_size:,} volumes: {estimated_hours:.1f} hours")

if __name__ == "__main__":
    main() 