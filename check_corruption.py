#!/usr/bin/env python3
"""
Check for data corruption between working (64³) vs broken (96³) datasets
"""

import argparse
import tarfile
import numpy as np
from pathlib import Path
import sys

def analyze_shard_stats(shard_path, volume_size, max_volumes=2000):
    """Analyze statistical properties of volumes in a shard."""
    print(f"Analyzing {shard_path.name}...")
    
    volume_count = 0
    stats = {
        'mins': [], 'maxs': [], 'means': [], 'stds': [],
        'extreme_count': 0, 'zero_count': 0, 'corrupt_count': 0
    }
    
    try:
        with tarfile.open(shard_path, "r|") as tar:
            for member in tar:
                if volume_count >= max_volumes:
                    break
                    
                try:
                    buf = tar.extractfile(member).read()
                    expected_bytes = volume_size ** 3 * 4
                    
                    if len(buf) != expected_bytes:
                        stats['corrupt_count'] += 1
                        print(f"  CORRUPT: {member.name} - {len(buf)} bytes (expected {expected_bytes})")
                        continue
                        
                    vol = np.frombuffer(buf, np.float32).reshape(volume_size, volume_size, volume_size)
                    
                    # Check for NaN/Inf
                    if np.isnan(vol).any() or np.isinf(vol).any():
                        stats['corrupt_count'] += 1
                        print(f"  NaN/INF: {member.name}")
                        continue
                    
                    # Compute stats
                    vol_min, vol_max = vol.min(), vol.max()
                    vol_mean, vol_std = vol.mean(), vol.std()
                    
                    stats['mins'].append(vol_min)
                    stats['maxs'].append(vol_max)
                    stats['means'].append(vol_mean)
                    stats['stds'].append(vol_std)
                    
                    # Check for extreme values (outside normal membrane range)
                    if vol_min < -1 or vol_max > 2:
                        stats['extreme_count'] += 1
                        if stats['extreme_count'] <= 3:  # Show first few
                            print(f"  EXTREME: {member.name} - range [{vol_min:.3f}, {vol_max:.3f}]")
                    
                    # Check for all-zero volumes
                    if np.all(vol == 0):
                        stats['zero_count'] += 1
                        print(f"  ALL-ZERO: {member.name}")
                    
                    volume_count += 1
                    
                except Exception as e:
                    stats['corrupt_count'] += 1
                    print(f"  EXCEPTION: {member.name} - {e}")
                    
    except Exception as e:
        print(f"  TAR ERROR: {e}")
        return None, volume_count
    
    # Convert to numpy arrays for stats
    for key in ['mins', 'maxs', 'means', 'stds']:
        if stats[key]:
            stats[key] = np.array(stats[key])
        else:
            stats[key] = np.array([])
    
    return stats, volume_count

def compare_datasets(dir_64, dir_96):
    """Compare characteristics of 64³ vs 96³ datasets."""
    
    print(f"64³ dataset: {dir_64}")
    print(f"96³ dataset: {dir_96}")
    
    # Get shard lists
    shards_64 = sorted(Path(dir_64).glob("shard*.tar"))[:5]  # First 5 shards
    shards_96 = sorted(Path(dir_96).glob("shard*.tar"))[:]  # First 5 shards
    
    print(f"\nFound {len(shards_64)} shards in 64³ dataset")
    print(f"Found {len(shards_96)} shards in 96³ dataset")
    
    # Analyze 64³ dataset
    print(f"\n{'='*60}")
    print("ANALYZING 64³ DATASET (WORKING)")
    print(f"{'='*60}")
    
    all_stats_64 = []
    for shard in shards_64:
        stats, count = analyze_shard_stats(shard, 64, 1000)
        if stats:
            all_stats_64.append((shard.name, stats, count))
    
    # Analyze 96³ dataset  
    print(f"\n{'='*60}")
    print("ANALYZING 96³ DATASET (BROKEN)")
    print(f"{'='*60}")
    
    all_stats_96 = []
    for shard in shards_96:
        stats, count = analyze_shard_stats(shard, 96, 1000)
        if stats:
            all_stats_96.append((shard.name, stats, count))
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print("\n64³ Dataset Summary:")
    total_corrupt_64 = sum(stats['corrupt_count'] for _, stats, _ in all_stats_64)
    total_extreme_64 = sum(stats['extreme_count'] for _, stats, _ in all_stats_64)
    total_volumes_64 = sum(count for _, _, count in all_stats_64)
    print(f"  Total volumes: {total_volumes_64}")
    print(f"  Corrupt volumes: {total_corrupt_64}")
    print(f"  Extreme values: {total_extreme_64}")
    
    if all_stats_64:
        all_mins_64 = np.concatenate([stats['mins'] for _, stats, _ in all_stats_64 if len(stats['mins']) > 0])
        all_maxs_64 = np.concatenate([stats['maxs'] for _, stats, _ in all_stats_64 if len(stats['maxs']) > 0])
        if len(all_mins_64) > 0:
            print(f"  Value range: [{all_mins_64.min():.3f}, {all_maxs_64.max():.3f}]")
            print(f"  Mean range: [{all_mins_64.mean():.3f}, {all_maxs_64.mean():.3f}]")
    
    print("\n96³ Dataset Summary:")
    total_corrupt_96 = sum(stats['corrupt_count'] for _, stats, _ in all_stats_96)
    total_extreme_96 = sum(stats['extreme_count'] for _, stats, _ in all_stats_96)
    total_volumes_96 = sum(count for _, _, count in all_stats_96)
    print(f"  Total volumes: {total_volumes_96}")
    print(f"  Corrupt volumes: {total_corrupt_96}")
    print(f"  Extreme values: {total_extreme_96}")
    
    if all_stats_96:
        all_mins_96 = np.concatenate([stats['mins'] for _, stats, _ in all_stats_96 if len(stats['mins']) > 0])
        all_maxs_96 = np.concatenate([stats['maxs'] for _, stats, _ in all_stats_96 if len(stats['maxs']) > 0])
        if len(all_mins_96) > 0:
            print(f"  Value range: [{all_mins_96.min():.3f}, {all_maxs_96.max():.3f}]")
            print(f"  Mean range: [{all_mins_96.mean():.3f}, {all_maxs_96.mean():.3f}]")
    
    # Check for significant differences
    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print(f"{'='*60}")
    
    if total_corrupt_96 > total_corrupt_64:
        print(f"🔍 FOUND ISSUE: 96³ has {total_corrupt_96} corrupt volumes vs {total_corrupt_64} in 64³")
        print("   This suggests the 96³ generation job was interrupted or had issues.")
    
    if total_extreme_96 > total_extreme_96 * 2:
        print(f"🔍 FOUND ISSUE: 96³ has {total_extreme_96} extreme values vs {total_extreme_64} in 64³")
        print("   Extreme values can cause numerical overflow during training.")
    
    if total_corrupt_96 == 0 and total_extreme_96 == 0:
        print("🤔 NO OBVIOUS CORRUPTION: Data looks statistically similar.")
        print("   The issue might be more subtle (partial volumes, etc.)")

def main():
    parser = argparse.ArgumentParser(description="Check for data corruption")
    parser.add_argument("--dir_64", default="/gpfs/radev/home/am3833/scratch/volumes_64")
    parser.add_argument("--dir_96", default="/gpfs/radev/home/am3833/scratch/volumes_96")
    args = parser.parse_args()
    
    # Check directories exist
    if not Path(args.dir_64).exists():
        print(f"ERROR: 64³ directory {args.dir_64} does not exist")
        sys.exit(1)
    if not Path(args.dir_96).exists():
        print(f"ERROR: 96³ directory {args.dir_96} does not exist")
        sys.exit(1)
    
    compare_datasets(args.dir_64, args.dir_96)

if __name__ == "__main__":
    main() 