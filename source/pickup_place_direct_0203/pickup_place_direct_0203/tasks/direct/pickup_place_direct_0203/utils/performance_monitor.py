# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Performance monitoring utility for tracking CPU/GPU bottlenecks during training.
"""

import torch
import time
from typing import Dict, Optional
from collections import deque
import statistics


class PerformanceMonitor:
    """Track timings and memory usage for various pipeline stages."""
    
    def __init__(self, window_size: int = 100, enabled: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of recent measurements to keep
            enabled: Whether to track performance
        """
        self.enabled = enabled
        self.window_size = window_size
        self.timings: Dict[str, deque] = {}
        self.memory_usage: Dict[str, float] = {}
        self.step_count = 0
        
        # GPU synchronization device (for accurate timing)
        self.device = None
        
    def set_device(self, device: torch.device):
        """Set the CUDA device for synchronization."""
        self.device = device
    
    def _sync_if_gpu(self):
        """Synchronize GPU if available for accurate timing."""
        if self.device is not None and str(self.device).startswith('cuda'):
            torch.cuda.synchronize(self.device)
    
    def start_timer(self, stage_name: str):
        """Start a timer for a named stage."""
        if not self.enabled:
            return
        
        self._sync_if_gpu()
        if not hasattr(self, f'_timer_{stage_name}'):
            setattr(self, f'_timer_{stage_name}', time.perf_counter())
        else:
            setattr(self, f'_timer_{stage_name}', time.perf_counter())
    
    def end_timer(self, stage_name: str):
        """End a timer and record the elapsed time."""
        if not self.enabled:
            return
        
        self._sync_if_gpu()
        start_time = getattr(self, f'_timer_{stage_name}', None)
        if start_time is None:
            return
        
        elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        if stage_name not in self.timings:
            self.timings[stage_name] = deque(maxlen=self.window_size)
        
        self.timings[stage_name].append(elapsed)
    
    def record_memory(self, stage_name: str, device_type: str = "cuda"):
        """Record peak memory usage."""
        if not self.enabled:
            return
        
        try:
            if device_type == "cuda" and torch.cuda.is_available():
                self.memory_usage[stage_name] = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            elif device_type == "cpu":
                import psutil
                process = psutil.Process()
                self.memory_usage[stage_name] = process.memory_info().rss / (1024**2)  # MB
        except Exception as e:
            print(f"[PerformanceMonitor] Warning: Failed to record memory: {e}")
    
    def log_summary(self, step: int, num_envs: int, prefix: str = ""):
        """Log performance summary at regular intervals."""
        if not self.enabled or not self.timings:
            return
        
        self.step_count = step
        
        summary_lines = [
            f"\n{'='*80}",
            f"[PERFORMANCE] Step {step}, Environments: {num_envs}",
            f"{'='*80}",
        ]
        
        # Calculate statistics for each stage
        total_time_ms = 0
        for stage_name in sorted(self.timings.keys()):
            times = list(self.timings[stage_name])
            if not times:
                continue
            
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            total_time_ms += avg_time
            
            # Percentage of total
            pct_str = ""
            if total_time_ms > 0:
                pct_str = f" ({avg_time/total_time_ms*100:.1f}%)"
            
            summary_lines.append(
                f"  {prefix}{stage_name:30s}: {avg_time:8.2f}ms [min:{min_time:7.2f}ms, max:{max_time:7.2f}ms]{pct_str}"
            )
        
        summary_lines.extend([
            f"{'-'*80}",
            f"  Total (all stages):             {total_time_ms:8.2f}ms",
        ])
        
        # Add memory info if available
        if self.memory_usage:
            summary_lines.append(f"{'-'*80}")
            for stage_name in sorted(self.memory_usage.keys()):
                summary_lines.append(
                    f"  Memory [{stage_name:25s}]: {self.memory_usage[stage_name]:8.2f} MB"
                )
        
        summary_lines.append(f"{'='*80}\n")
        
        print('\n'.join(summary_lines))
        
        return total_time_ms
    
    def get_average(self, stage_name: str) -> Optional[float]:
        """Get average time for a stage in milliseconds."""
        if stage_name not in self.timings or not self.timings[stage_name]:
            return None
        
        return statistics.mean(list(self.timings[stage_name]))
    
    def get_bottleneck(self) -> Optional[str]:
        """Identify the slowest stage."""
        if not self.timings:
            return None
        
        slowest_stage = None
        slowest_time = 0
        
        for stage_name, times in self.timings.items():
            if times:
                avg_time = statistics.mean(list(times))
                if avg_time > slowest_time:
                    slowest_time = avg_time
                    slowest_stage = stage_name
        
        return f"{slowest_stage} ({slowest_time:.2f}ms)" if slowest_stage else None
    
    def reset(self):
        """Reset all timings."""
        self.timings = {}
        self.memory_usage = {}


# Global performance monitor instance
_global_perf_monitor = None


def get_perf_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor."""
    global _global_perf_monitor
    if _global_perf_monitor is None:
        _global_perf_monitor = PerformanceMonitor(enabled=True)
    return _global_perf_monitor
