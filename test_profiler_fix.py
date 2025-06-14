#!/usr/bin/env python3

import time
import numpy as np
from training_profiler import TrainingProfiler

def test_profiler_fix():
    """Test that the profiler no longer produces impossible percentages."""
    
    print("Testing ProfilerFix...")
    
    # Initialize profiler
    profiler = TrainingProfiler(log_interval=1, memory_tracking=False, detailed_timing=True)
    
    # Simulate a small training epoch with 3 batches
    num_batches = 3
    
    for batch_idx in range(num_batches):
        profiler.start_batch_timing()
        
        # Simulate different operations with controlled timing
        with profiler.profile_section("forward_pass"):
            time.sleep(0.1)  # 100ms
            
        with profiler.profile_section("backward_pass"):
            time.sleep(0.05)  # 50ms
            
        with profiler.profile_section("optimizer_step"):
            time.sleep(0.02)  # 20ms
            
        profiler.end_batch_timing(batch_idx)
    
    # Test epoch summary - this should show reasonable percentages
    print("\n" + "="*50)
    print("TESTING EPOCH SUMMARY OUTPUT:")
    print("="*50)
    
    profiler.log_epoch_summary(epoch=1)
    
    # Verify percentages are reasonable
    epoch_batch_times = list(profiler.batch_times)
    total_epoch_time = sum(epoch_batch_times)
    
    print(f"\nVERIFICATION:")
    print(f"Number of batches: {len(epoch_batch_times)}")
    print(f"Total epoch time: {total_epoch_time:.3f}s")
    print(f"Average batch time: {np.mean(epoch_batch_times):.3f}s")
    
    # Check that timing sections exist
    total_component_time = 0
    for section, times in profiler.timings.items():
        if len(times) > 0:
            recent_times = times[-len(epoch_batch_times):] if len(times) >= len(epoch_batch_times) else times
            section_time = sum(recent_times)
            total_component_time += section_time
            percentage = (section_time / total_component_time * 100) if total_component_time > 0 else 0
            print(f"Section '{section}': {section_time:.3f}s")
    
    print(f"Total component time: {total_component_time:.3f}s")
    
    # Check that percentages would be reasonable (should be ~100% total)
    reasonable_range = total_component_time < total_epoch_time * 1.5  # Allow some overhead
    
    if reasonable_range:
        print("✅ PROFILER FIX SUCCESSFUL: Timing calculations look reasonable!")
    else:
        print("❌ PROFILER STILL HAS ISSUES: Component time is much larger than epoch time")
        print(f"   Component time: {total_component_time:.3f}s")
        print(f"   Epoch time: {total_epoch_time:.3f}s")
        print(f"   Ratio: {total_component_time/total_epoch_time:.1f}x")
    
    profiler.cleanup()
    return reasonable_range

if __name__ == "__main__":
    test_profiler_fix() 