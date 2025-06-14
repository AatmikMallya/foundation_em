# MAE Training Profiling System

## Overview
The integrated profiling system provides comprehensive performance monitoring for your MAE training pipeline, helping identify bottlenecks and optimize performance.

## Metrics Tracked in WandB

### 🕒 **Timing Metrics**
- `profiling/batch_time_avg` - Average time per training batch
- `profiling/batch_time_std` - Standard deviation of batch times
- `profiling/batches_per_second` - Training throughput
- `profiling/epoch_total_time` - Total time per epoch
- `profiling/epoch_throughput_batches_per_hour` - Epoch-level throughput

### 📊 **Detailed Timing Breakdown**
- `profiling/timing_data_transfer_avg` - GPU data transfer time
- `profiling/timing_forward_pass_avg` - Model forward pass time
- `profiling/timing_backward_pass_avg` - Gradient computation time
- `profiling/timing_optimizer_step_avg` - Optimizer update time
- `profiling/timing_ema_update_avg` - EMA model update time
- `profiling/timing_validation_epoch_avg` - Validation time
- `profiling/timing_visualization_generation_avg` - Visualization generation time

### 📈 **Epoch-Level Timing Percentages**
- `profiling/epoch_forward_pass_percentage` - % of epoch time spent on forward pass
- `profiling/epoch_backward_pass_percentage` - % of epoch time spent on backward pass
- `profiling/epoch_data_transfer_percentage` - % of epoch time spent on data transfer
- `profiling/epoch_optimizer_step_percentage` - % of epoch time spent on optimizer steps
- `profiling/epoch_validation_epoch_percentage` - % of epoch time spent on validation
- `profiling/epoch_ema_update_percentage` - % of epoch time spent on EMA updates
- `profiling/epoch_visualization_generation_percentage` - % of epoch time spent on visualization
- `profiling/epoch_other_percentage` - % of epoch time unaccounted for (overhead)

### 💾 **Memory Usage**
- `profiling/ram_usage_percent` - System RAM usage percentage
- `profiling/ram_available_gb` - Available RAM in GB
- `profiling/gpu_0_memory_allocated_gb` - GPU memory allocated
- `profiling/gpu_0_memory_utilization` - GPU memory utilization percentage
- `profiling/epoch_max_ram_usage` - Peak RAM usage per epoch
- `profiling/epoch_max_gpu_0_memory` - Peak GPU memory per epoch

### 🖥️ **System Resources**
- `profiling/cpu_usage_percent` - CPU utilization
- `profiling/data_generation_time` - Time spent generating synthetic data
- `profiling/data_transfer_time` - Time transferring data to GPU
- `profiling/data_gen_percentage` - Percentage of batch time spent on data generation
- `profiling/transfer_percentage` - Percentage of batch time spent on data transfer

### 🚨 **Bottleneck Detection**
- `profiling/bottleneck_data_generation` - Boolean: Is data generation a bottleneck?
- `profiling/data_gen_bottleneck_severity` - Severity of data generation bottleneck (0-1)
- `profiling/bottleneck_memory` - Boolean: Is memory usage too high?
- `profiling/memory_pressure` - Memory pressure level (0-1)
- `profiling/bottleneck_gpu_0_memory` - Boolean: Is GPU memory a bottleneck?
- `profiling/gpu_0_memory_pressure` - GPU memory pressure (0-1)

## Console Output

### Per-Epoch Summary
```
=== Epoch 1 Profiling Summary ===
Total time: 45.23s
Avg batch time: 0.142s
Throughput: 2547.3 batches/hour

📊 Time Breakdown:
  forward_pass             : 45.2% (20.45s)
  backward_pass            : 28.1% (12.71s)
  data_transfer            :  8.3% ( 3.75s)
  optimizer_step           :  7.8% ( 3.53s)
  validation_epoch         :  6.2% ( 2.80s)
  ema_update              :  2.1% ( 0.95s)
  visualization_generation :  1.8% ( 0.81s)
  other/overhead          :  0.5% ( 0.23s)
==================================================
```

### Bottleneck Warnings
- ⚠️ **Data generation may be a bottleneck!** (>30% of batch time)
- ⚠️ **High memory usage detected!** (>85% RAM usage)

## Optimization Insights

### 🔍 **Data Generation Bottleneck**
If `profiling/data_gen_percentage > 30%`:
- **Solution**: Increase `num_workers` in DataLoader
- **Solution**: Use faster data generation (pre-computed vs on-the-fly)
- **Solution**: Optimize synthetic data generation algorithms

### 🔍 **Memory Bottleneck**
If `profiling/memory_pressure > 0.85`:
- **Solution**: Reduce batch size
- **Solution**: Use gradient accumulation instead of larger batches
- **Solution**: Enable memory-efficient training options

### 🔍 **GPU Memory Bottleneck**
If `profiling/gpu_0_memory_pressure > 0.90`:
- **Solution**: Reduce batch size or model size
- **Solution**: Use gradient checkpointing
- **Solution**: Enable mixed precision training (AMP)

### 🔍 **Data Transfer Bottleneck**
If `profiling/transfer_percentage > 10%`:
- **Solution**: Use `pin_memory=True` in DataLoader
- **Solution**: Increase `num_workers` for parallel data loading
- **Solution**: Use faster storage (SSD vs HDD)

## WandB Dashboard Views

### 📈 **Performance Dashboard**
Create custom charts in WandB:
1. **Throughput Over Time**: `profiling/batches_per_second` vs epoch
2. **Memory Usage**: `profiling/ram_usage_percent` and `profiling/gpu_0_memory_utilization`
3. **Timing Breakdown**: Stacked area chart of different timing components
4. **Bottleneck Detection**: Boolean charts showing when bottlenecks occur

### 📊 **Optimization Tracking**
Monitor improvements after changes:
- Compare `profiling/epoch_throughput_batches_per_hour` before/after optimizations
- Track `profiling/data_gen_percentage` when tuning data loading
- Monitor `profiling/memory_pressure` when adjusting batch sizes

## Integration Details

### Automatic Profiling
The profiler automatically tracks:
- ✅ Every training batch (timing, memory)
- ✅ Data generation and transfer
- ✅ Forward/backward passes
- ✅ Optimizer steps and EMA updates
- ✅ Validation loops
- ✅ Visualization generation

### Background Monitoring
Continuous monitoring (1-second intervals):
- CPU usage percentage
- RAM usage and availability
- GPU memory allocation and utilization
- System resource pressure

### Minimal Overhead
- Profiling overhead: <1% of training time
- Background monitoring: Separate thread, minimal impact
- Automatic cleanup: Resources freed at training end

## Usage Tips

1. **Monitor Early**: Check profiling metrics in first few epochs
2. **Compare Configurations**: Use profiling to A/B test different settings
3. **Identify Trends**: Look for degrading performance over time
4. **Optimize Iteratively**: Make one change at a time and measure impact
5. **Set Alerts**: Use WandB alerts for bottleneck detection metrics

## Example Optimization Workflow

1. **Baseline**: Run training with profiling enabled
2. **Identify**: Check which `profiling/timing_*` metrics are highest
3. **Optimize**: Make targeted improvements (e.g., increase `num_workers`)
4. **Measure**: Compare new `profiling/epoch_throughput_batches_per_hour`
5. **Iterate**: Repeat until satisfactory performance

The profiling system provides actionable insights to maximize your training efficiency! 🚀 