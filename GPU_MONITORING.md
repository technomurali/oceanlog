# GPU Monitoring for Nava Ocean Log

This guide explains how to monitor GPU utilization during container processing with your Nava Ocean Log application.

## Quick Start

### 1. Install Dependencies

```bash
pip install nvidia-ml-py3 psutil
```

### 2. Easy Monitoring (Recommended)

```bash
# Use the interactive script
./monitor_gpu.sh
```

### 3. Manual Monitoring

```bash
# Monitor all GPU processes
python3 gpu_utilization.py --all-processes

# Monitor specific process (get PID from application logs)
python3 gpu_utilization.py --pid 12345
```

## When to Use GPU Monitoring

Monitor GPU usage when:
- Running container detection with YOLOv8 models
- Processing large video files
- Debugging performance issues
- Optimizing processing parameters

## Integration with Your Application

### Modified Files

The following files now display the process PID when GPU processing starts:

1. **`UI/model_inference/agent_container.py`**
   - Shows PID when container back detection starts
   - Displays GPU name and device information

2. **`UI/model_inference/agent_container_video_macker.py`**
   - Shows PID for video processing functions
   - Displays device (CPU/GPU) information

### Example Output

When you run your application, you'll see:

```
[INIT] Running on GPU: NVIDIA GeForce RTX 3080
[INIT] Process PID: 12345 - Use this PID for GPU monitoring
```

Use this PID for targeted monitoring.

## Monitoring Options

### 1. Real-time Monitoring

```bash
# Monitor all GPU activity (recommended)
python3 gpu_utilization.py --all-processes

# Monitor specific process
python3 gpu_utilization.py --pid 12345

# Update every 1 second
python3 gpu_utilization.py --all-processes --interval 1
```

### 2. CSV Logging

```bash
# Log GPU usage to CSV file for analysis
python3 gpu_utilization.py --all-processes --csv gpu_log.csv
```

### 3. Limited Duration Monitoring

```bash
# Monitor for 10 iterations only
python3 gpu_utilization.py --all-processes --iterations 10
```

## Understanding the Output

### Real-time Display

```
=== GPU Status at 14:32:15 ===

GPU 0 (NVIDIA GeForce RTX 3080):
  Core Utilization: 85%
  Memory: 6147 / 10240 MiB (60.0%)
  Temperature: 72°C
  Power Usage: 245W
  Active Processes:
    • PID 12345 (python3) – 1500 MiB VRAM
```

### Key Metrics

- **Core Utilization**: GPU compute usage (0-100%)
- **Memory**: GPU memory usage in MiB and percentage
- **Temperature**: GPU temperature in Celsius
- **Power Usage**: Current power consumption in Watts
- **Active Processes**: List of processes using the GPU

## Usage Scenarios

### Scenario 1: Processing a Video File

1. **Start the monitoring** (in a separate terminal):
   ```bash
   ./monitor_gpu.sh
   # Choose option 1 (Monitor all GPU processes)
   ```

2. **Run your application**:
   ```bash
   cd UI
   streamlit run app.py --server.fileWatcherType none
   ```

3. **Upload and process a video** through the web interface

4. **Watch the GPU metrics** in the monitoring terminal

### Scenario 2: Debugging Performance Issues

1. **Start CSV logging**:
   ```bash
   python3 gpu_utilization.py --all-processes --csv performance_test.csv
   ```

2. **Run your processing pipeline**

3. **Analyze the CSV file** to identify bottlenecks

### Scenario 3: Monitoring Specific Process

1. **Start your application and note the PID** from the logs

2. **Monitor only that process**:
   ```bash
   python3 gpu_utilization.py --pid <YOUR_PID>
   ```

## Troubleshooting

### Common Issues

#### "nvidia-ml-py3 package is missing"
```bash
pip install nvidia-ml-py3
```

#### "nvidia-smi not found"
- Ensure NVIDIA drivers are installed
- Check if you're on a GPU-enabled system

#### "No GPU activity detected"
- Your application might be using CPU instead of GPU
- Check the application logs for device information

#### Permission Issues
```bash
# Make the monitoring script executable
chmod +x monitor_gpu.sh
```

### Verifying GPU Setup

```bash
# Check if GPU is detected
nvidia-smi

# Test GPU access from Python
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Quick GPU status
python3 gpu_utilization.py --all-processes --iterations 1
```

## Performance Optimization Tips

### Based on GPU Monitoring

1. **Low GPU Utilization (<50%)**
   - Increase batch size if possible
   - Check if CPU is the bottleneck
   - Verify GPU acceleration is enabled

2. **High Memory Usage (>90%)**
   - Reduce batch size
   - Process smaller video chunks
   - Clear GPU memory between operations

3. **High Temperature (>80°C)**
   - Improve cooling
   - Reduce processing intensity
   - Monitor for thermal throttling

4. **Power Usage Concerns**
   - Monitor power consumption trends
   - Consider power limits for extended processing

## Advanced Usage

### Custom Monitoring Scripts

You can integrate the monitoring functions into your own scripts:

```python
from gpu_utilization import get_gpu_stats

# Get current GPU statistics
stats = get_gpu_stats()
for gpu in stats:
    print(f"GPU {gpu['gpu_index']}: {gpu['gpu_util']}% utilization")
```

### Automated Alerts

```bash
# Monitor and alert if GPU utilization drops below 50%
while true; do
    util=$(python3 -c "
from gpu_utilization import get_gpu_stats
stats = get_gpu_stats()
print(stats[0]['gpu_util'] if stats else 0)
")
    if [ "$util" -lt 50 ]; then
        echo "Alert: Low GPU utilization: ${util}%"
    fi
    sleep 10
done
```

## Files Overview

- **`gpu_utilization.py`**: Main GPU monitoring script
- **`monitor_gpu.sh`**: Interactive monitoring script
- **`GPU_MONITORING.md`**: This documentation file
- **Modified application files**:
  - `UI/model_inference/agent_container.py`
  - `UI/model_inference/agent_container_video_macker.py`

## Next Steps

1. **Test the monitoring** with a sample video
2. **Establish baseline metrics** for your typical workloads
3. **Set up automated logging** for production monitoring
4. **Optimize your processing** based on the metrics

For support or questions, refer to the main project documentation or create an issue in the repository. 