#!/usr/bin/env python3
"""
gpu_utilization.py
Realtime GPU-utilization monitor for a single process (or all processes).

Prerequisites
-------------
$ pip install nvidia-ml-py3

Usage examples
--------------
# Monitor the process whose PID is 12345 every 2 s, indefinitely
$ python gpu_utilization.py --pid 12345

# Monitor *this* Python interpreter (default) 5 times at 1 s intervals
$ python gpu_utilization.py --iterations 5 --interval 1
"""
import argparse
import os
import time
import sys

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
      - gpu_util             (%)
      - memory_util          (%)
      - processes            (list of {'pid', 'used_memory_mb'})
    If pid_filter is set, only that PID’s usage is reported.
    """
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    results = []

    for idx in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # Active compute processes on this GPU
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except pynvml.NVMLError:
            procs = []

        proc_stats = []
        for p in procs:
            if pid_filter is None or p.pid == pid_filter:
                proc_stats.append(
                    {"pid": p.pid, "used_memory_mb": p.usedGpuMemory // 1024 // 1024}
                )

        results.append(
            {
                "gpu_index": idx,
                "gpu_util": util.gpu,  # %
                "memory_util": 100 * mem.used / mem.total,  # %
                "processes": proc_stats,
            }
        )

    pynvml.nvmlShutdown()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor GPU utilization.")
    parser.add_argument(
        "--pid",
        type=int,
        default=os.getpid(),
        help="PID to monitor (default: this Python process)",
    )
    parser.add_argument(
        "--interval", type=float, default=2.0, help="Seconds between updates"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=-1,
        help="Number of updates (-1 = run indefinitely)",
    )
    args = parser.parse_args()

    iteration = 0
    while args.iterations < 0 or iteration < args.iterations:
        stats = get_gpu_stats(args.pid)
        print(f"\n=== iteration {iteration + 1} (PID {args.pid}) ===")
        any_output = False
        for s in stats:
            if not s["processes"]:
                continue
            any_output = True
            print(
                f"GPU {s['gpu_index']}: "
                f"{s['gpu_util']}% core util, {s['memory_util']:.1f}% memory"
            )
            for proc in s["processes"]:
                print(f"  • PID {proc['pid']} – {proc['used_memory_mb']} MiB VRAM")

        if not any_output:
            print("No matching GPU activity for the specified PID.")

        iteration += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()