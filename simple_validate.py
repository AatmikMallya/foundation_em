#!/usr/bin/env python3
"""
Simple Dataset Validation Script
===============================
Checks tar shard datasets for NaN/Inf values and corruption.
"""

import argparse
import tarfile
import numpy as np
import sys
from pathlib import Path

def validate_shard(shard_path, volume_size, max_volumes=1000):
    """Validate a single tar shard file."""
    print(f"Validating {shard_path.name}...")
    
    errors = []
    warnings = []
    volume_count = 0
    nan_volumes = []
    inf_volumes = []
    
    try:
        with tarfile.open(shard_path, "r|") as tar:
            for member in tar:
                if volume_count >= max_volumes:
                    break
                    
                try:
                    # Extract volume data
                    buf = tar.extractfile(member).read()
                    expected_bytes = volume_size ** 3 * 4  # float32 = 4 bytes
                    
                    if len(buf) != expected_bytes:
                        errors.append(f"Volume {member.name}: Expected {expected_bytes} bytes, got {len(buf)}")
                        continue
                        
                    # Parse as float32 volume
                    vol = np.frombuffer(buf, np.float32).reshape(volume_size, volume_size, volume_size)
                    
                    # Check for NaN/Inf values
                    nan_count = np.isnan(vol).sum()
                    inf_count = np.isinf(vol).sum()
                    
                    if nan_count > 0:
                        nan_volumes.append((member.name, nan_count))
                        errors.append(f"Volume {member.name}: Contains {nan_count} NaN values")
                    
                    if inf_count > 0:
                        inf_volumes.append((member.name, inf_count))
                        errors.append(f"Volume {member.name}: Contains {inf_count} Inf values")
                    
                    # Check value range
                    vol_min, vol_max = vol.min(), vol.max()
                    if vol_min < -10 or vol_max > 10:
                        warnings.append(f"Volume {member.name}: Range [{vol_min:.3f}, {vol_max:.3f}]")
                    
                    volume_count += 1
                    
                    # Progress update
                    if volume_count % 1000 == 0:
                        print(f"  Checked {volume_count} volumes...")
                        
                except Exception as e:
                    errors.append(f"Volume {member.name}: Exception - {e}")
                    
    except Exception as e:
        errors.append(f"Tar file error: {e}")
        return errors, warnings, volume_count, nan_volumes, inf_volumes
    
    print(f"  Completed: {volume_count} volumes checked")
    return errors, warnings, volume_count, nan_volumes, inf_volumes

def main():
    parser = argparse.ArgumentParser(description="Simple dataset validation")
    parser.add_argument("--shard_dir", required=True)
    parser.add_argument("--volume_size", type=int, default=96)
    parser.add_argument("--max_shards", type=int, default=5)
    parser.add_argument("--max_volumes_per_shard", type=int, default=5000)
    args = parser.parse_args()
    
    shard_dir = Path(args.shard_dir)
    if not shard_dir.exists():
        print(f"ERROR: {shard_dir} does not exist")
        sys.exit(1)
    
    shards = sorted(shard_dir.glob("shard*.tar"))[:args.max_shards]
    if not shards:
        print(f"ERROR: No shard files found in {shard_dir}")
        sys.exit(1)
    
    print(f"Found {len(shards)} shard files to validate")
    print(f"Volume size: {args.volume_size}³")
    print(f"Max volumes per shard: {args.max_volumes_per_shard}")
    
    total_errors = 0
    total_warnings = 0
    total_volumes = 0
    total_nan_volumes = 0
    total_inf_volumes = 0
    
    for shard_path in shards:
        errors, warnings, volume_count, nan_vols, inf_vols = validate_shard(
            shard_path, args.volume_size, args.max_volumes_per_shard
        )
        
        total_errors += len(errors)
        total_warnings += len(warnings)
        total_volumes += volume_count
        total_nan_volumes += len(nan_vols)
        total_inf_volumes += len(inf_vols)
        
        if errors:
            print(f"❌ {shard_path.name}: {len(errors)} errors")
            for error in errors[:3]:  # Show first 3 errors
                print(f"  ERROR: {error}")
            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more errors")
        elif warnings:
            print(f"⚠️  {shard_path.name}: {len(warnings)} warnings")
        else:
            print(f"✅ {shard_path.name}: OK")
        
        if nan_vols:
            print(f"  NaN volumes found: {len(nan_vols)}")
            for vol_name, count in nan_vols[:3]:
                print(f"    {vol_name}: {count} NaN values")
        
        if inf_vols:
            print(f"  Inf volumes found: {len(inf_vols)}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Shards checked: {len(shards)}")
    print(f"Total volumes: {total_volumes}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")
    print(f"Volumes with NaN: {total_nan_volumes}")
    print(f"Volumes with Inf: {total_inf_volumes}")
    
    if total_errors > 0:
        print("\n❌ DATASET HAS ERRORS - This could cause NaN during training!")
        sys.exit(1)
    elif total_nan_volumes > 0 or total_inf_volumes > 0:
        print("\n⚠️ DATASET HAS NaN/Inf VALUES - This WILL cause NaN during training!")
        sys.exit(1)
    else:
        print("\n✅ DATASET VALIDATION PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main() 