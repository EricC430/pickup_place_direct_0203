#!/usr/bin/env python3
"""
Quick GPU Memory Usage Diagnostic Script

Helps identify GPU memory consumption before and after optimization.

Usage:
    python check_gpu_memory.py
"""

import subprocess
import time
import sys


def get_gpu_memory_info():
    """Get current GPU memory usage using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return None
        
        parts = result.stdout.strip().split(", ")
        return {
            "used_mb": float(parts[0]),
            "total_mb": float(parts[1]),
            "free_mb": float(parts[2])
        }
    except Exception as e:
        print(f"[ERROR] Failed to get GPU memory info: {e}")
        return None


def print_memory_header():
    """Print header for memory monitoring."""
    print("\n" + "="*70)
    print("GPU MEMORY USAGE DIAGNOSTIC")
    print("="*70)


def monitor_memory(label: str, duration: int = 2):
    """Monitor GPU memory usage for a duration."""
    print(f"\n[{label}] Monitoring GPU memory for {duration} seconds...")
    
    memory_samples = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        info = get_gpu_memory_info()
        if info:
            memory_samples.append(info)
        time.sleep(0.5)
    
    if not memory_samples:
        print("[ERROR] Could not get GPU memory info")
        return None
    
    # Analyze samples
    used_values = [s["used_mb"] for s in memory_samples]
    max_used = max(used_values)
    min_used = min(used_values)
    avg_used = sum(used_values) / len(used_values)
    
    print(f"  Memory Used: {min_used:.0f}MB - {max_used:.0f}MB (avg: {avg_used:.0f}MB)")
    print(f"  Total Memory: {memory_samples[0]['total_mb']:.0f}MB")
    print(f"  Free Memory: {min(s['free_mb'] for s in memory_samples):.0f}MB")
    
    return max_used


def main():
    """Main diagnostic routine."""
    print_memory_header()
    
    # Check if nvidia-smi is available
    try:
        result = subprocess.run(["nvidia-smi", "--version"], capture_output=True, timeout=2)
        if result.returncode != 0:
            print("[ERROR] nvidia-smi not found. Please ensure NVIDIA drivers are installed.")
            return 1
    except Exception as e:
        print(f"[ERROR] Cannot run nvidia-smi: {e}")
        return 1
    
    print("\n[INFO] GPU Memory Diagnostic Tool")
    print("-" * 70)
    
    # Get baseline memory usage
    baseline_info = get_gpu_memory_info()
    if not baseline_info:
        print("[ERROR] Cannot access GPU memory information")
        return 1
    
    print(f"\n[BASELINE] Current GPU Memory Status:")
    print(f"  Used: {baseline_info['used_mb']:.0f}MB / {baseline_info['total_mb']:.0f}MB")
    print(f"  Free: {baseline_info['free_mb']:.0f}MB")
    print(f"  Available: {baseline_info['free_mb'] / baseline_info['total_mb'] * 100:.1f}%")
    
    # Check if there's enough memory for interactive debugging
    print("\n[CHECK] Memory Sufficiency for Interactive Debug:")
    print(f"  Required for optimized script: ~150-200MB")
    print(f"  Current available: {baseline_info['free_mb']:.0f}MB")
    
    if baseline_info['free_mb'] < 150:
        print(f"  ⚠️  WARNING: Low available GPU memory ({baseline_info['free_mb']:.0f}MB)")
        print(f"     Consider closing other GPU applications")
    elif baseline_info['free_mb'] < 400:
        print(f"  ⚠️  CAUTION: Limited GPU memory available ({baseline_info['free_mb']:.0f}MB)")
        print(f"     Optimization strongly recommended")
    else:
        print(f"  ✓ Sufficient memory available for optimized script")
    
    # Recommendations
    print("\n[RECOMMENDATIONS] Before running interactive_vision_debug.py:")
    print("  1. Close other GPU applications (TensorFlow, PyTorch notebooks, etc.)")
    print("  2. Run: nvidia-smi to verify available memory")
    print("  3. Verify headless=True in SimulationContext")
    print("  4. Verify YOLO device='cpu' in config")
    print("  5. If still failing, try killing background processes:")
    print("     - Kill Jupyter kernels")
    print("     - Kill lingering Python processes")
    print("     - Reboot if necessary")
    
    # Memory optimization checklist
    print("\n[OPTIMIZATION CHECKLIST]:")
    optimizations = [
        ("headless=True (disable graphics)", "Check interactive_vision_debug.py line ~140"),
        ("yolo_device='cpu' (CPU inference)", "Check pickup_place_direct_0203_vision_asym_env_cfg.py line ~78"),
        ("yolo_model_name='yolov8n' (nano model)", "Check same config file"),
        ("Single camera only (camera_high=None)", "Check interactive_vision_debug.py line ~235"),
    ]
    
    for i, (opt_name, location) in enumerate(optimizations, 1):
        print(f"  [{i}] {opt_name}")
        print(f"      {location}")
    
    print("\n" + "="*70)
    print("Run this script again after starting the interactive vision debug")
    print("to monitor GPU memory consumption:")
    print("  python check_gpu_memory.py")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
