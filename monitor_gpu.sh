#!/bin/bash

# monitor_gpu.sh - Easy GPU monitoring for Nava Ocean Log
# This script helps you monitor GPU utilization during container processing

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Nava Ocean Log GPU Monitor ===${NC}"
echo

# Check if nvidia-ml-py3 is installed
if ! python3 -c "import pynvml" 2>/dev/null; then
    echo -e "${RED}Error: nvidia-ml-py3 is not installed${NC}"
    echo -e "${YELLOW}Install it with: pip install nvidia-ml-py3${NC}"
    exit 1
fi

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. No NVIDIA GPU detected.${NC}"
    exit 1
fi

echo -e "${GREEN}GPU detected! Available options:${NC}"
echo
echo "1. Monitor all GPU processes (recommended)"
echo "2. Monitor specific PID"
echo "3. Monitor with CSV logging"
echo "4. Quick GPU status check"
echo

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo -e "${BLUE}Monitoring all GPU processes. Press Ctrl+C to stop.${NC}"
        python3 gpu_utilization.py --all-processes
        ;;
    2)
        read -p "Enter PID to monitor: " pid
        echo -e "${BLUE}Monitoring PID $pid. Press Ctrl+C to stop.${NC}"
        python3 gpu_utilization.py --pid $pid
        ;;
    3)
        timestamp=$(date +"%Y%m%d_%H%M%S")
        csv_file="gpu_log_${timestamp}.csv"
        echo -e "${BLUE}Monitoring with CSV logging to $csv_file. Press Ctrl+C to stop.${NC}"
        python3 gpu_utilization.py --all-processes --csv $csv_file
        ;;
    4)
        echo -e "${BLUE}Current GPU Status:${NC}"
        python3 gpu_utilization.py --all-processes --iterations 1
        ;;
    *)
        echo -e "${RED}Invalid option. Exiting.${NC}"
        exit 1
        ;;
esac 