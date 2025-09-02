#!/usr/bin/env python3
"""
gpu_utilization.py
Real-time GPU utilization monitor for Nava Ocean Log container processing.

Prerequisites
-------------
$ pip install nvidia-ml-py3

Usage examples
--------------
# Monitor the process whose PID is 12345 every 2 s, indefinitely
$ python gpu_utilization.py --pid 12345

# Monitor *this* Python interpreter (default) 5 times at 1 s intervals
$ python gpu_utilization.py --iterations 5 --interval 1

# Monitor all GPU processes (no PID filter)
$ python gpu_utilization.py --all-processes
"""
import argparse
import os
import time
import sys
from datetime import datetime

try:
    import pynvml
except ImportError:
    sys.exit(
        "The 'nvidia-ml-py3' package is missing. "
        "Install it with:  pip install nvidia-ml-py3"
    )


def get_gpu_stats(pid_filter: int | None = None) -> list[dict]:
    """
    Return a list of dictionaries, one per GPU, containing:
      - gpu_index            (int)
      - gpu_name             (str)
      - gpu_util             (%)
      - memory_util          (%)
      - memory_used_mb       (int)
      - memory_total_mb      (int)
      - temperature          (°C)
      - power_usage          (W)
      - processes            (list of {'pid', 'used_memory_mb', 'process_name'})
    If pid_filter is set, only that PID's usage is reported.
    """
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    results = []

    for idx in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        
        # Basic GPU info
        gpu_name_raw = pynvml.nvmlDeviceGetName(handle)
        gpu_name = gpu_name_raw.decode('utf-8') if isinstance(gpu_name_raw, bytes) else gpu_name_raw
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # Temperature and power
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            temp = -1
            
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert to watts
        except pynvml.NVMLError:
            power = -1

        # Active compute processes on this GPU
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except pynvml.NVMLError:
            procs = []

        proc_stats = []
        for p in procs:
            if pid_filter is None or p.pid == pid_filter:
                # Try to get process name
                try:
                    import psutil
                    process = psutil.Process(p.pid)
                    process_name = process.name()
                except (ImportError, psutil.NoSuchProcess):
                    process_name = "unknown"
                
                proc_stats.append({
                    "pid": p.pid,
                    "used_memory_mb": p.usedGpuMemory // 1024 // 1024,
                    "process_name": process_name
                })

        results.append({
            "gpu_index": idx,
            "gpu_name": gpu_name,
            "gpu_util": util.gpu,  # %
            "memory_util": 100 * mem.used / mem.total,  # %
            "memory_used_mb": mem.used // 1024 // 1024,
            "memory_total_mb": mem.total // 1024 // 1024,
            "temperature": temp,
            "power_usage": power,
            "processes": proc_stats,
        })

    pynvml.nvmlShutdown()
    return results


def print_gpu_stats(stats: list[dict], show_all_processes: bool = False) -> bool:
    """
    Print GPU statistics in a formatted way.
    Returns True if any relevant processes were found, False otherwise.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n=== GPU Status at {timestamp} ===")
    
    any_output = False
    
    for s in stats:
        gpu_info = f"GPU {s['gpu_index']} ({s['gpu_name']})"
        
        if show_all_processes or s["processes"]:
            any_output = True
            print(f"\n{gpu_info}:")
            print(f"  Core Utilization: {s['gpu_util']}%")
            print(f"  Memory: {s['memory_used_mb']} / {s['memory_total_mb']} MiB ({s['memory_util']:.1f}%)")
            
            if s['temperature'] != -1:
                print(f"  Temperature: {s['temperature']}°C")
            if s['power_usage'] != -1:
                print(f"  Power Usage: {s['power_usage']}W")
            
            if s["processes"]:
                print(f"  Active Processes:")
                for proc in s["processes"]:
                    print(f"    • PID {proc['pid']} ({proc['process_name']}) – {proc['used_memory_mb']} MiB VRAM")
            elif show_all_processes:
                print(f"  No active processes")
    
    return any_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor GPU utilization for Nava Ocean Log processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gpu_utilization.py --pid 12345          # Monitor specific process
  python gpu_utilization.py --all-processes      # Monitor all GPU activity
  python gpu_utilization.py --interval 1         # Update every 1 second
  python gpu_utilization.py --iterations 10      # Run for 10 iterations only
        """
    )
    
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="PID to monitor (default: auto-detect from this process if not --all-processes)",
    )
    parser.add_argument(
        "--all-processes",
        action="store_true",
        help="Monitor all GPU processes instead of filtering by PID"
    )
    parser.add_argument(
        "--interval", 
        type=float, 
        default=2.0, 
        help="Seconds between updates (default: 2.0)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=-1,
        help="Number of updates (-1 = run indefinitely)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Save results to CSV file for analysis"
    )
    
    args = parser.parse_args()
    
    # Determine monitoring mode
    if args.all_processes:
        monitor_pid = None
        print(f"Monitoring ALL GPU processes (update every {args.interval}s)")
    elif args.pid:
        monitor_pid = args.pid
        print(f"Monitoring GPU usage for PID {monitor_pid} (update every {args.interval}s)")
    else:
        monitor_pid = os.getpid()
        print(f"Monitoring GPU usage for current process PID {monitor_pid} (update every {args.interval}s)")
    
    # CSV logging setup
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, 'w')
        csv_file.write("timestamp,gpu_index,gpu_util,memory_util,memory_used_mb,temperature,power_usage,processes\n")
    
    try:
        iteration = 0
        while args.iterations < 0 or iteration < args.iterations:
            stats = get_gpu_stats(monitor_pid)
            any_output = print_gpu_stats(stats, args.all_processes)
            
            # CSV logging
            if csv_file:
                timestamp = datetime.now().isoformat()
                for s in stats:
                    processes_str = f"{len(s['processes'])} processes" if s['processes'] else "no processes"
                    csv_file.write(f"{timestamp},{s['gpu_index']},{s['gpu_util']},{s['memory_util']:.1f},"
                                 f"{s['memory_used_mb']},{s['temperature']},{s['power_usage']},{processes_str}\n")
                csv_file.flush()
            
            if not any_output and not args.all_processes:
                print("No GPU activity detected for the specified process.")
            
            iteration += 1
            
            if args.iterations < 0 or iteration < args.iterations:
                time.sleep(args.interval)
                
    except KeyboardInterrupt:
        print("\n\nGPU monitoring stopped by user.")
    finally:
        if csv_file:
            csv_file.close()
            print(f"CSV data saved to: {args.csv}")


if __name__ == "__main__":
    main() 