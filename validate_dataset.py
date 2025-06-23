#!/usr/bin/env python3
"""
Dataset Validation Script
========================
Validates tar shard datasets for MAE training to detect:
- Corrupted tar files
- Incomplete tar files  
- Malformed volume data
- NaN/Inf values in volumes
- Missing expected files
- Dataloader iteration issues
"""

import argparse
import tarfile
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys
import traceback
from collections import defaultdict

def validate_single_shard(shard_path, volume_size, expected_count=65536):
    """Validate a single tar shard file."""
    errors = []
    warnings = []
    volume_count = 0
    
    try:
        with tarfile.open(shard_path, "r|") as tar:
            for member in tar:
                try:
                    # Extract volume data
                    buf = tar.extractfile(member).read()
                    expected_bytes = volume_size ** 3 * 4  # float32 = 4 bytes
                    
                    if len(buf) != expected_bytes:
                        errors.append(f"Volume {member.name}: Expected {expected_bytes} bytes, got {len(buf)}")
                        continue
                        
                    # Parse as float32 volume
                    try:
                        vol = np.frombuffer(buf, np.float32).reshape(volume_size, volume_size, volume_size)
                    except ValueError as e:
                        errors.append(f"Volume {member.name}: Failed to reshape - {e}")
                        continue
                    
                    # Check for NaN/Inf values
                    nan_count = np.isnan(vol).sum()
                    inf_count = np.isinf(vol).sum()
                    
                    if nan_count > 0:
                        errors.append(f"Volume {member.name}: Contains {nan_count} NaN values")
                    if inf_count > 0:
                        errors.append(f"Volume {member.name}: Contains {inf_count} Inf values")
                    
                    # Check value range sanity
                    if vol.min() < -10 or vol.max() > 10:
                        warnings.append(f"Volume {member.name}: Unusual range [{vol.min():.3f}, {vol.max():.3f}]")
                    
                    # Check for all-zero volumes
                    if np.all(vol == 0):
                        warnings.append(f"Volume {member.name}: All-zero volume")
                    
                    volume_count += 1
                    
                except Exception as e:
                    errors.append(f"Volume {member.name}: Exception during processing - {e}")
                    
    except tarfile.ReadError as e:
        errors.append(f"Tar file corrupted: {e}")
        return errors, warnings, 0
    except Exception as e:
        errors.append(f"Unexpected error reading tar: {e}")
        return errors, warnings, 0
    
    # Check volume count
    if volume_count != expected_count:
        if volume_count == 0:
            errors.append(f"No volumes found (expected {expected_count})")
        elif volume_count < expected_count:
            warnings.append(f"Incomplete shard: {volume_count}/{expected_count} volumes")
        else:
            warnings.append(f"Extra volumes: {volume_count}/{expected_count} volumes")
    
    return errors, warnings, volume_count

def validate_dataloader_iteration(shard_dir, volume_size, batch_size=16, max_batches=50):
    """Test that the dataloader can iterate without errors."""
    from vol_train import TarShardDataset
    from torch.utils.data import DataLoader
    
    print(f"\n=== Testing DataLoader Iteration ===")
    
    shards = sorted(Path(shard_dir).expanduser().glob("shard*.tar"))
    if not shards:
        print("No shard files found!")
        return False
    
    dataset = TarShardDataset(shards, volume_size, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, drop_last=True)
    
    errors = []
    nan_batches = []
    
    try:
        for i, batch in enumerate(tqdm(loader, desc="Testing batches", total=max_batches)):
            if i >= max_batches:
                break
                
            # Check batch for NaN/Inf
            if torch.isnan(batch).any():
                nan_count = torch.isnan(batch).sum().item()
                nan_batches.append((i, nan_count))
                
            if torch.isinf(batch).any():
                inf_count = torch.isinf(batch).sum().item()
                errors.append(f"Batch {i}: Contains {inf_count} Inf values")
            
            # Check batch shape
            expected_shape = (batch_size, 1, volume_size, volume_size, volume_size)
            if batch.shape != expected_shape:
                errors.append(f"Batch {i}: Wrong shape {batch.shape}, expected {expected_shape}")
    
    except Exception as e:
        errors.append(f"DataLoader iteration failed: {e}")
        traceback.print_exc()
        return False
    
    if errors:
        print("DataLoader validation FAILED:")
        for error in errors:
            print(f"  ERROR: {error}")
    
    if nan_batches:
        print("NaN values found in batches:")
        for batch_idx, count in nan_batches:
            print(f"  Batch {batch_idx}: {count} NaN values")
        return False
    
    if not errors and not nan_batches:
        print("DataLoader validation PASSED")
        return True
    
    return len(errors) == 0

def main():
    parser = argparse.ArgumentParser(description="Validate tar shard dataset integrity")
    parser.add_argument("--shard_dir", required=True, help="Directory containing shard*.tar files")
    parser.add_argument("--volume_size", type=int, default=96, help="Expected volume size (default: 96)")
    parser.add_argument("--expected_per_shard", type=int, default=65536, help="Expected volumes per shard")
    parser.add_argument("--test_dataloader", action="store_true", help="Test DataLoader iteration")
    parser.add_argument("--max_shards", type=int, default=None, help="Max number of shards to validate")
    args = parser.parse_args()
    
    shard_dir = Path(args.shard_dir).expanduser()
    if not shard_dir.exists():
        print(f"ERROR: Shard directory {shard_dir} does not exist")
        sys.exit(1)
    
    shards = sorted(shard_dir.glob("shard*.tar"))
    if not shards:
        print(f"ERROR: No shard*.tar files found in {shard_dir}")
        sys.exit(1)
    
    print(f"Found {len(shards)} shard files")
    print(f"Expected volume size: {args.volume_size}³")
    print(f"Expected volumes per shard: {args.expected_per_shard}")
    
    if args.max_shards:
        shards = shards[:args.max_shards]
        print(f"Validating first {len(shards)} shards only")
    
    # Validate each shard
    total_errors = 0
    total_warnings = 0
    total_volumes = 0
    failed_shards = []
    
    print(f"\n=== Validating {len(shards)} Shards ===")
    
    for shard_path in tqdm(shards, desc="Validating shards"):
        try:
            errors, warnings, volume_count = validate_single_shard(
                shard_path, args.volume_size, args.expected_per_shard
            )
            
            total_errors += len(errors)
            total_warnings += len(warnings)
            total_volumes += volume_count
            
            if errors:
                failed_shards.append(shard_path.name)
                print(f"\n❌ {shard_path.name}: {len(errors)} errors, {len(warnings)} warnings")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"  ERROR: {error}")
                if len(errors) > 5:
                    print(f"  ... and {len(errors) - 5} more errors")
            elif warnings:
                print(f"\n⚠️  {shard_path.name}: {len(warnings)} warnings")
                for warning in warnings[:3]:  # Show first 3 warnings
                    print(f"  WARNING: {warning}")
                if len(warnings) > 3:
                    print(f"  ... and {len(warnings) - 3} more warnings")
            else:
                print(f"✅ {shard_path.name}: OK ({volume_count} volumes)")
                
        except Exception as e:
            failed_shards.append(shard_path.name)
            total_errors += 1
            print(f"\n💥 {shard_path.name}: CRASHED - {e}")
            traceback.print_exc()
    
    # Summary
    print(f"\n=== Validation Summary ===")
    print(f"Total shards checked: {len(shards)}")
    print(f"Failed shards: {len(failed_shards)}")
    print(f"Total volumes: {total_volumes}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")
    
    if failed_shards:
        print(f"\nFailed shards: {', '.join(failed_shards)}")
    
    # Test dataloader if requested
    dataloader_ok = True
    if args.test_dataloader:
        dataloader_ok = validate_dataloader_iteration(shard_dir, args.volume_size)
    
    # Final verdict
    if total_errors == 0 and dataloader_ok:
        print("\n🎉 DATASET VALIDATION PASSED")
        sys.exit(0)
    else:
        print(f"\n❌ DATASET VALIDATION FAILED")
        print(f"   Errors: {total_errors}")
        if not dataloader_ok:
            print("   DataLoader test failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 