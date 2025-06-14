#!/usr/bin/env python3

import time
import psutil
import torch
import wandb
import numpy as np
from collections import defaultdict, deque
from contextlib import contextmanager
import threading
import gc

class TrainingProfiler:
    """Comprehensive profiler for MAE training to identify bottlenecks and track performance."""
    
    def __init__(self, log_interval=10, memory_tracking=True, detailed_timing=True):
        self.log_interval = log_interval
        self.memory_tracking = memory_tracking
        self.detailed_timing = detailed_timing
        
        # Timing storage
        self.timings = defaultdict(list)
        self.current_timings = {}
        self.batch_times = deque(maxlen=100)  # Rolling window
        
        # Memory tracking
        self.memory_stats = defaultdict(list)
        self.gpu_memory_stats = defaultdict(list)
        
        # Data generation tracking
        self.data_gen_times = deque(maxlen=50)
        self.data_transfer_times = deque(maxlen=50)
        self.data_generation_lock = threading.Lock()
        
        # System info
        self.cpu_count = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        
        # GPU info
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        else:
            self.gpu_count = 0
            self.gpu_memory_total = 0
            
        # Background monitoring
        self.monitoring = False
        self.monitor_thread = None
        
        print(f"TrainingProfiler initialized:")
        print(f"  CPU cores: {self.cpu_count}")
        print(f"  Total RAM: {self.total_memory:.1f} GB")
        print(f"  GPU count: {self.gpu_count}")
        if self.gpu_count > 0:
            print(f"  GPU memory: {self.gpu_memory_total:.1f} GB")
    
    @contextmanager
    def profile_section(self, section_name):
        """Context manager for timing specific sections."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = {k: end_memory[k] - start_memory[k] for k in start_memory.keys()}
            
            self.timings[section_name].append(duration)
            if self.memory_tracking:
                self.memory_stats[f"{section_name}_memory_delta"].append(memory_delta)
    
    def start_batch_timing(self):
        """Start timing a training batch."""
        self.batch_start_time = time.time()
        self.batch_start_memory = self._get_memory_usage()
        
        # Start background monitoring if not already running
        if not self.monitoring and self.memory_tracking:
            self.start_background_monitoring()
    
    def end_batch_timing(self, batch_idx):
        """End timing a training batch and log statistics."""
        batch_duration = time.time() - self.batch_start_time
        self.batch_times.append(batch_duration)
        
        if self.memory_tracking:
            end_memory = self._get_memory_usage()
            memory_delta = {k: end_memory[k] - self.batch_start_memory[k] for k in self.batch_start_memory.keys()}
            self.memory_stats["batch_memory_delta"].append(memory_delta)
        
        # Log to WandB at intervals
        if batch_idx % self.log_interval == 0:
            self._log_batch_stats(batch_idx)
    
    def profile_data_generation(self, data_gen_func, *args, **kwargs):
        """Profile data generation/loading time."""
        start_time = time.time()
        result = data_gen_func(*args, **kwargs)
        end_time = time.time()
        
        self.data_gen_times.append(end_time - start_time)
        return result
    
    def profile_data_transfer(self, data, device):
        """Profile data transfer to GPU time."""
        start_time = time.time()
        data_on_device = data.to(device)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        self.data_transfer_times.append(end_time - start_time)
        return data_on_device
    
    def add_data_generation_time(self, generation_time):
        """Add a data generation time measurement from the dataset."""
        with self.data_generation_lock:
            self.data_gen_times.append(generation_time)
    
    def start_background_monitoring(self):
        """Start background monitoring of system resources."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_background_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _background_monitor(self):
        """Background thread for continuous monitoring."""
        while self.monitoring:
            try:
                # CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory_info = psutil.virtual_memory()
                
                self.memory_stats["cpu_percent"].append(cpu_percent)
                self.memory_stats["ram_percent"].append(memory_info.percent)
                self.memory_stats["ram_available_gb"].append(memory_info.available / (1024**3))
                
                # GPU monitoring
                if torch.cuda.is_available():
                    for gpu_id in range(self.gpu_count):
                        gpu_memory = torch.cuda.memory_stats(gpu_id)
                        allocated_gb = gpu_memory.get('allocated_bytes.all.current', 0) / (1024**3)
                        reserved_gb = gpu_memory.get('reserved_bytes.all.current', 0) / (1024**3)
                        
                        self.gpu_memory_stats[f"gpu_{gpu_id}_allocated_gb"].append(allocated_gb)
                        self.gpu_memory_stats[f"gpu_{gpu_id}_reserved_gb"].append(reserved_gb)
                        self.gpu_memory_stats[f"gpu_{gpu_id}_utilization"].append(
                            allocated_gb / self.gpu_memory_total * 100
                        )
                
            except Exception as e:
                print(f"Background monitoring error: {e}")
                break
    
    def _get_memory_usage(self):
        """Get current memory usage statistics."""
        stats = {}
        
        # System memory
        memory_info = psutil.virtual_memory()
        stats['ram_used_gb'] = memory_info.used / (1024**3)
        stats['ram_percent'] = memory_info.percent
        
        # GPU memory
        if torch.cuda.is_available():
            for gpu_id in range(self.gpu_count):
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                stats[f'gpu_{gpu_id}_allocated_gb'] = allocated
                stats[f'gpu_{gpu_id}_reserved_gb'] = reserved
        
        return stats
    
    def _log_batch_stats(self, batch_idx):
        """Log batch statistics to WandB."""
        if len(self.batch_times) == 0:
            return
            
        # Timing statistics
        recent_batch_times = list(self.batch_times)
        avg_batch_time = np.mean(recent_batch_times)
        std_batch_time = np.std(recent_batch_times)
        
        log_dict = {
            "profiling/batch_time_avg": avg_batch_time,
            "profiling/batch_time_std": std_batch_time,
            "profiling/batches_per_second": 1.0 / avg_batch_time if avg_batch_time > 0 else 0,
        }
        
        # Data generation timing
        if len(self.data_gen_times) > 0:
            avg_data_gen = np.mean(list(self.data_gen_times))
            log_dict["profiling/data_generation_time"] = avg_data_gen
            log_dict["profiling/data_gen_percentage"] = (avg_data_gen / avg_batch_time * 100) if avg_batch_time > 0 else 0
        
        # Data transfer timing
        if len(self.data_transfer_times) > 0:
            avg_transfer = np.mean(list(self.data_transfer_times))
            log_dict["profiling/data_transfer_time"] = avg_transfer
            log_dict["profiling/transfer_percentage"] = (avg_transfer / avg_batch_time * 100) if avg_batch_time > 0 else 0
        
        # Memory statistics
        if self.memory_tracking and len(self.memory_stats["ram_percent"]) > 0:
            log_dict["profiling/ram_usage_percent"] = self.memory_stats["ram_percent"][-1]
            log_dict["profiling/cpu_usage_percent"] = self.memory_stats["cpu_percent"][-1] if len(self.memory_stats["cpu_percent"]) > 0 else 0
            
            if len(self.memory_stats["ram_available_gb"]) > 0:
                log_dict["profiling/ram_available_gb"] = self.memory_stats["ram_available_gb"][-1]
        
        # GPU statistics
        if torch.cuda.is_available() and len(self.gpu_memory_stats) > 0:
            for gpu_id in range(self.gpu_count):
                if f"gpu_{gpu_id}_allocated_gb" in self.gpu_memory_stats:
                    if len(self.gpu_memory_stats[f"gpu_{gpu_id}_allocated_gb"]) > 0:
                        log_dict[f"profiling/gpu_{gpu_id}_memory_allocated_gb"] = self.gpu_memory_stats[f"gpu_{gpu_id}_allocated_gb"][-1]
                        log_dict[f"profiling/gpu_{gpu_id}_memory_utilization"] = self.gpu_memory_stats[f"gpu_{gpu_id}_utilization"][-1]
        
        # Detailed timing breakdown
        if self.detailed_timing:
            for section, times in self.timings.items():
                if len(times) > 0:
                    recent_times = times[-10:]  # Last 10 measurements
                    log_dict[f"profiling/timing_{section}_avg"] = np.mean(recent_times)
                    if len(recent_times) > 1:
                        log_dict[f"profiling/timing_{section}_std"] = np.std(recent_times)
        
        # Bottleneck analysis
        bottleneck_analysis = self._analyze_bottlenecks()
        log_dict.update(bottleneck_analysis)
        
        wandb.log(log_dict, step=batch_idx)
    
    def _analyze_bottlenecks(self):
        """Analyze potential bottlenecks in the training pipeline."""
        analysis = {}
        
        if len(self.batch_times) == 0:
            return analysis
            
        avg_batch_time = np.mean(list(self.batch_times))
        
        # Data generation bottleneck
        if len(self.data_gen_times) > 0:
            avg_data_gen = np.mean(list(self.data_gen_times))
            data_gen_ratio = avg_data_gen / avg_batch_time
            analysis["profiling/bottleneck_data_generation"] = data_gen_ratio > 0.3  # >30% is concerning
            analysis["profiling/data_gen_bottleneck_severity"] = min(data_gen_ratio, 1.0)
        
        # Memory bottleneck
        if len(self.memory_stats["ram_percent"]) > 0:
            ram_usage = self.memory_stats["ram_percent"][-1]
            analysis["profiling/bottleneck_memory"] = ram_usage > 85  # >85% RAM usage
            analysis["profiling/memory_pressure"] = ram_usage / 100.0
        
        # GPU memory bottleneck
        if torch.cuda.is_available() and len(self.gpu_memory_stats) > 0:
            for gpu_id in range(self.gpu_count):
                if f"gpu_{gpu_id}_utilization" in self.gpu_memory_stats:
                    if len(self.gpu_memory_stats[f"gpu_{gpu_id}_utilization"]) > 0:
                        gpu_util = self.gpu_memory_stats[f"gpu_{gpu_id}_utilization"][-1]
                        analysis[f"profiling/bottleneck_gpu_{gpu_id}_memory"] = gpu_util > 90  # >90% GPU memory
                        analysis[f"profiling/gpu_{gpu_id}_memory_pressure"] = gpu_util / 100.0
        
        return analysis
    
    def log_epoch_summary(self, epoch):
        """Log comprehensive epoch summary with detailed timing breakdown."""
        if len(self.batch_times) == 0:
            return
            
        # Calculate epoch statistics
        epoch_batch_times = list(self.batch_times)
        total_epoch_time = sum(epoch_batch_times)
        avg_batch_time = np.mean(epoch_batch_times)
        
        summary = {
            "profiling/epoch_total_time": total_epoch_time,
            "profiling/epoch_avg_batch_time": avg_batch_time,
            "profiling/epoch_batches_processed": len(epoch_batch_times),
            "profiling/epoch_throughput_batches_per_hour": len(epoch_batch_times) / (total_epoch_time / 3600) if total_epoch_time > 0 else 0,
        }
        
        # Detailed timing breakdown with percentages
        timing_breakdown = {}
        timing_percentages = {}
        total_component_time = 0
        
        # First pass: calculate total component time for this epoch only
        for section, times in self.timings.items():
            if len(times) > 0:
                # Sum only the timing measurements from this epoch's batches
                recent_times = times[-len(epoch_batch_times):] if len(times) >= len(epoch_batch_times) else times
                section_epoch_time = sum(recent_times)
                total_component_time += section_epoch_time
        
        # Second pass: calculate percentages and log metrics
        for section, times in self.timings.items():
            if len(times) > 0:
                # Use only recent timing measurements for this epoch
                recent_times = times[-len(epoch_batch_times):] if len(times) >= len(epoch_batch_times) else times
                section_epoch_time = sum(recent_times)
                section_avg_time = np.mean(recent_times)
                # Calculate percentage against actual epoch time spent in components
                section_percentage = (section_epoch_time / total_component_time * 100) if total_component_time > 0 else 0
                
                timing_breakdown[f"profiling/epoch_{section}_total_time"] = section_epoch_time
                timing_breakdown[f"profiling/epoch_{section}_avg_time"] = section_avg_time
                timing_percentages[f"profiling/epoch_{section}_percentage"] = section_percentage
        
        # Add timing breakdown to summary
        summary.update(timing_breakdown)
        summary.update(timing_percentages)
        
        # Calculate "other" time (unaccounted time)
        # Note: total_component_time now represents time spent in profiled sections
        # Other time would be total_epoch_time - total_component_time, but since 
        # total_component_time may be larger (overlapping sections), we skip "other" 
        # calculation to avoid negative values
        if total_component_time < total_epoch_time:
            other_time = total_epoch_time - total_component_time
            summary["profiling/epoch_other_time"] = other_time
            # Calculate other percentage relative to total component time for consistency
            summary["profiling/epoch_other_percentage"] = (other_time / total_component_time * 100) if total_component_time > 0 else 0
        
        # Data generation summary
        if len(self.data_gen_times) > 0:
            # Use recent data generation times for this epoch
            recent_data_gen_times = list(self.data_gen_times)[-len(epoch_batch_times):] if len(self.data_gen_times) >= len(epoch_batch_times) else list(self.data_gen_times)
            epoch_data_gen_time = sum(recent_data_gen_times)
            summary["profiling/epoch_data_generation_time"] = epoch_data_gen_time
            summary["profiling/epoch_data_gen_percentage"] = (epoch_data_gen_time / total_epoch_time * 100) if total_epoch_time > 0 else 0
        
        # Memory summary
        if len(self.memory_stats["ram_percent"]) > 0:
            summary["profiling/epoch_max_ram_usage"] = max(self.memory_stats["ram_percent"])
            summary["profiling/epoch_avg_ram_usage"] = np.mean(self.memory_stats["ram_percent"])
        
        # GPU summary
        if torch.cuda.is_available():
            for gpu_id in range(self.gpu_count):
                if f"gpu_{gpu_id}_utilization" in self.gpu_memory_stats:
                    if len(self.gpu_memory_stats[f"gpu_{gpu_id}_utilization"]) > 0:
                        summary[f"profiling/epoch_max_gpu_{gpu_id}_memory"] = max(self.gpu_memory_stats[f"gpu_{gpu_id}_utilization"])
                        summary[f"profiling/epoch_avg_gpu_{gpu_id}_memory"] = np.mean(self.gpu_memory_stats[f"gpu_{gpu_id}_utilization"])
        
        wandb.log(summary, step=epoch)
        
        # Print detailed summary to console
        print(f"\n=== Epoch {epoch} Profiling Summary ===")
        print(f"Total time: {total_epoch_time:.2f}s")
        print(f"Avg batch time: {avg_batch_time:.3f}s")
        print(f"Throughput: {len(epoch_batch_times) / (total_epoch_time / 3600):.1f} batches/hour")
        
        # Print timing breakdown
        print("\n📊 Time Breakdown:")
        timing_items = []
        
        # Calculate total component time for this epoch only
        epoch_component_time = 0
        for section, times in self.timings.items():
            if len(times) > 0:
                # Sum only the timing measurements from this epoch's batches
                recent_times = times[-len(epoch_batch_times):] if len(times) >= len(epoch_batch_times) else times
                section_epoch_time = sum(recent_times)
                epoch_component_time += section_epoch_time
        
        for section, times in self.timings.items():
            if len(times) > 0:
                # Use only recent timing measurements for this epoch
                recent_times = times[-len(epoch_batch_times):] if len(times) >= len(epoch_batch_times) else times
                section_epoch_time = sum(recent_times)
                # Calculate percentage against actual epoch time spent in components
                section_percentage = (section_epoch_time / epoch_component_time * 100) if epoch_component_time > 0 else 0
                timing_items.append((section, section_percentage, section_epoch_time))
        
        # Sort by percentage (highest first)
        timing_items.sort(key=lambda x: x[1], reverse=True)
        
        for section, percentage, total_time in timing_items:
            print(f"  {section:25s}: {percentage:5.1f}% ({total_time:6.2f}s)")
        
        # Display other/overhead time if it exists
        if epoch_component_time < total_epoch_time:
            other_time = total_epoch_time - epoch_component_time
            other_percentage = (other_time / epoch_component_time * 100) if epoch_component_time > 0 else 0
            print(f"  {'other/overhead':25s}: {other_percentage:5.1f}% ({other_time:6.2f}s)")
        
        # Data generation warning
        if len(self.data_gen_times) > 0:
            recent_data_gen_times = list(self.data_gen_times)[-len(epoch_batch_times):] if len(self.data_gen_times) >= len(epoch_batch_times) else list(self.data_gen_times)
            epoch_data_gen_time = sum(recent_data_gen_times)
            data_gen_pct = (epoch_data_gen_time / total_epoch_time * 100) if total_epoch_time > 0 else 0
            if data_gen_pct > 30:
                print(f"\n⚠️  Data generation: {data_gen_pct:.1f}% - may be a bottleneck!")
        
        # Memory warning
        if len(self.memory_stats["ram_percent"]) > 0:
            max_ram = max(self.memory_stats["ram_percent"])
            if max_ram > 85:
                print(f"⚠️  Peak RAM usage: {max_ram:.1f}% - high memory usage!")
        
        print("=" * 50)
    
    def cleanup(self):
        """Clean up profiler resources."""
        self.stop_background_monitoring()
        
        # Clear large data structures
        self.timings.clear()
        self.memory_stats.clear()
        self.gpu_memory_stats.clear()
        self.batch_times.clear()
        self.data_gen_times.clear()
        self.data_transfer_times.clear()

# Context manager for easy profiling integration
@contextmanager
def profile_training_step(profiler, step_name):
    """Context manager for profiling individual training steps."""
    with profiler.profile_section(step_name):
        yield

# Decorator for profiling functions
def profile_function(profiler, section_name=None):
    """Decorator to automatically profile function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = section_name or f"{func.__module__}.{func.__name__}"
            with profiler.profile_section(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator 