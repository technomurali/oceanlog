# Nava Ocean Log - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Nava Ocean Log system in various environments, from development to production.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Development Environment](#development-environment)
- [Production Deployment](#production-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Performance Optimization](#performance-optimization)
- [Monitoring and Logging](#monitoring-and-logging)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Ubuntu 18.04+ / CentOS 7+ / Windows 10+
- **CPU**: 4 cores, 2.4 GHz
- **RAM**: 8GB
- **Storage**: 20GB free space
- **GPU**: Optional (CPU-only processing supported)

#### Recommended Requirements
- **OS**: Ubuntu 20.04+ / CentOS 8+
- **CPU**: 8+ cores, 3.0 GHz
- **RAM**: 16GB+
- **Storage**: 50GB+ SSD
- **GPU**: NVIDIA RTX 3080+ (8GB+ VRAM)

### Software Dependencies

#### Core Dependencies
```bash
# Python 3.8+
python3 --version

# CUDA Toolkit (for GPU acceleration)
nvidia-smi

# FFmpeg (for video processing)
ffmpeg -version

# Git
git --version
```

#### Python Dependencies
```bash
# Core packages
pip install streamlit>=1.28.0
pip install opencv-python>=4.8.0
pip install torch>=2.0.0
pip install ultralytics>=8.0.0
pip install paddleocr>=2.7.0
pip install pandas>=2.0.0

# Development packages
pip install black isort pytest
```

## Development Environment

### Local Setup

#### 1. Clone Repository
```bash
git clone <repository-url>
cd navaoceanlog
```

#### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

#### 3. Install Dependencies
```bash
# Install requirements
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

#### 4. Download Models
```bash
# Create models directory
mkdir -p UI/models

# Download pre-trained models (replace with actual URLs)
wget -O UI/models/container_back_identifier.pt <model-url-1>
wget -O UI/models/container_back_labeled.pt <model-url-2>
```

#### 5. Verify Installation
```bash
# Test GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test model loading
python -c "from ultralytics import YOLO; model = YOLO('UI/models/container_back_identifier.pt')"

# Test OCR
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR()"
```

#### 6. Run Development Server
```bash
cd UI
streamlit run app.py --server.fileWatcherType none
```

### Development Configuration

#### Environment Variables
```bash
# Create .env file
cat > .env << EOF
# Development settings
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# Model paths
CONTAINER_BACK_MODEL=UI/models/container_back_identifier.pt
CONTAINER_LABEL_MODEL=UI/models/container_back_labeled.pt

# Processing settings
CONFIDENCE_THRESHOLD=0.953
PROGRESS_FREQUENCY=25
EOF
```

#### Development Scripts
```bash
# Create development scripts
cat > scripts/dev.sh << 'EOF'
#!/bin/bash
# Development startup script
cd UI
export $(cat ../.env | xargs)
streamlit run app.py --server.fileWatcherType none
EOF

chmod +x scripts/dev.sh
```

## Production Deployment

### System Preparation

#### 1. System Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv nginx ffmpeg
```

#### 2. User Setup
```bash
# Create application user
sudo useradd -m -s /bin/bash navaocean
sudo usermod -aG sudo navaocean

# Switch to application user
sudo su - navaocean
```

#### 3. Application Directory
```bash
# Create application directory
mkdir -p /home/navaocean/app
cd /home/navaocean/app

# Clone application
git clone <repository-url> .
```

### Production Configuration

#### 1. Environment Setup
```bash
# Create production environment
python3 -m venv venv
source venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
```

#### 2. Production Configuration
```bash
# Create production config
cat > config/production.py << EOF
import os

# Production settings
DEBUG = False
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Server settings
HOST = '0.0.0.0'
PORT = 8501
WORKERS = 4

# Model settings
MODEL_PATH = '/home/navaocean/app/UI/models'
UPLOAD_PATH = '/home/navaocean/app/UI/uploaded_video_files'
OUTPUT_PATH = '/home/navaocean/app/UI/video_outputs'

# Security settings
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = ['mp4']
EOF
```

#### 3. Systemd Service
```bash
# Create systemd service file
sudo tee /etc/systemd/system/navaocean.service << EOF
[Unit]
Description=Nava Ocean Log
After=network.target

[Service]
Type=simple
User=navaocean
WorkingDirectory=/home/navaocean/app/UI
Environment=PATH=/home/navaocean/app/venv/bin
ExecStart=/home/navaocean/app/venv/bin/streamlit run app.py --server.fileWatcherType none --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable navaocean
sudo systemctl start navaocean
```

#### 4. Nginx Configuration
```bash
# Create Nginx configuration
sudo tee /etc/nginx/sites-available/navaocean << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }

    # File upload size limit
    client_max_body_size 500M;
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/navaocean /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Docker Deployment

### Dockerfile
```dockerfile
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p UI/uploaded_video_files UI/video_outputs UI/models

# Expose port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run application
CMD ["streamlit", "run", "UI/app.py", "--server.fileWatcherType", "none"]
EOF
```

### Docker Compose
```yaml
# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  navaocean:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./UI/uploaded_video_files:/app/UI/uploaded_video_files
      - ./UI/video_outputs:/app/UI/video_outputs
      - ./UI/models:/app/UI/models
    environment:
      - STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - navaocean
    restart: unless-stopped
EOF
```

### Docker Deployment Commands
```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

## Cloud Deployment

### AWS EC2 Deployment

#### 1. Launch EC2 Instance
```bash
# Launch GPU instance (g4dn.xlarge or larger)
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type g4dn.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --count 1
```

#### 2. Instance Setup
```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv nginx ffmpeg

# Install NVIDIA drivers (if using GPU instance)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
```

#### 3. Application Deployment
```bash
# Clone application
git clone <repository-url> /home/ubuntu/navaoceanlog
cd /home/ubuntu/navaoceanlog

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download models
mkdir -p UI/models
# Download your models here

# Run application
cd UI
streamlit run app.py --server.fileWatcherType none --server.port 8501 --server.address 0.0.0.0
```

### Google Cloud Platform

#### 1. Create VM Instance
```bash
# Create GPU instance
gcloud compute instances create navaocean-instance \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --boot-disk-size=50GB \
    --maintenance-policy=TERMINATE \
    --restart-on-failure
```

#### 2. Setup Application
```bash
# Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv nginx ffmpeg

# Install NVIDIA drivers
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Deploy application
git clone <repository-url> /home/your-user/navaoceanlog
cd /home/your-user/navaoceanlog
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Azure Deployment

#### 1. Create VM
```bash
# Create resource group
az group create --name navaocean-rg --location eastus

# Create VM with GPU
az vm create \
    --resource-group navaocean-rg \
    --name navaocean-vm \
    --image UbuntuLTS \
    --size Standard_NC6s_v3 \
    --admin-username azureuser \
    --generate-ssh-keys
```

#### 2. Deploy Application
```bash
# Connect to VM
ssh azureuser@your-vm-ip

# Install dependencies and deploy application
# (Similar to AWS/GCP setup)
```

## Performance Optimization

### GPU Optimization

#### 1. CUDA Configuration
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
```

#### 2. Model Optimization
```python
# Optimize model loading
import torch
from ultralytics import YOLO

# Load model with optimizations
model = YOLO('model.pt')
model.to('cuda')
model.fuse()  # Fuse layers for faster inference
```

### Memory Management

#### 1. Batch Processing
```python
# Process videos in batches
def process_video_batch(video_paths, batch_size=4):
    for i in range(0, len(video_paths), batch_size):
        batch = video_paths[i:i+batch_size]
        # Process batch
        torch.cuda.empty_cache()  # Clear GPU memory
```

#### 2. Resource Monitoring
```bash
# Monitor system resources
htop
nvidia-smi -l 1  # GPU monitoring
iotop  # I/O monitoring
```

### Scaling Strategies

#### 1. Horizontal Scaling
```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: navaocean
spec:
  replicas: 3
  selector:
    matchLabels:
      app: navaocean
  template:
    metadata:
      labels:
        app: navaocean
    spec:
      containers:
      - name: navaocean
        image: navaocean:latest
        ports:
        - containerPort: 8501
        resources:
          limits:
            nvidia.com/gpu: 1
```

#### 2. Load Balancing
```nginx
# Nginx load balancer configuration
upstream navaocean_backend {
    server 127.0.0.1:8501;
    server 127.0.0.1:8502;
    server 127.0.0.1:8503;
}

server {
    listen 80;
    location / {
        proxy_pass http://navaocean_backend;
    }
}
```

## Monitoring and Logging

### Application Monitoring

#### 1. Logging Configuration
```python
# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('navaocean.log'),
        logging.StreamHandler()
    ]
)
```

#### 2. Performance Metrics
```python
# Track processing metrics
import time
import psutil

def track_performance():
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    # Processing code here
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    processing_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    logging.info(f"Processing time: {processing_time:.2f}s")
    logging.info(f"Memory used: {memory_used / 1024 / 1024:.2f}MB")
```

### System Monitoring

#### 1. Prometheus Metrics
```python
# Export metrics for Prometheus
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
PROCESSING_TIME = Histogram('video_processing_seconds', 'Time spent processing videos')
VIDEOS_PROCESSED = Counter('videos_processed_total', 'Total videos processed')

# Start metrics server
start_http_server(8000)
```

#### 2. Health Checks
```python
# Health check endpoint
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0'
    }
```

## Security Considerations

### 1. File Upload Security
```python
# Validate uploaded files
import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'mp4'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file(file):
    if file.content_length > MAX_FILE_SIZE:
        raise ValueError("File too large")
    
    if not allowed_file(file.filename):
        raise ValueError("Invalid file type")
    
    filename = secure_filename(file.filename)
    return filename
```

### 2. Authentication
```python
# Basic authentication
import streamlit as st

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 User not known or password incorrect")
        return False
    else:
        return True
```

### 3. HTTPS Configuration
```nginx
# SSL configuration
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8501;
        # ... other proxy settings
    }
}
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size in model configuration
```

#### 2. Model Loading Errors
```bash
# Check model file integrity
ls -la UI/models/
md5sum UI/models/*.pt

# Verify model compatibility
python -c "from ultralytics import YOLO; model = YOLO('UI/models/container_back_identifier.pt')"
```

#### 3. Service Not Starting
```bash
# Check service status
sudo systemctl status navaocean

# View service logs
sudo journalctl -u navaocean -f

# Check port availability
sudo netstat -tlnp | grep 8501
```

#### 4. Performance Issues
```bash
# Monitor system resources
htop
iostat -x 1
nvidia-smi -l 1

# Check disk space
df -h
du -sh UI/video_outputs/
```

### Debug Mode
```bash
# Run in debug mode
cd UI
STREAMLIT_DEBUG=1 streamlit run app.py --server.fileWatcherType none

# Enable verbose logging
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/navaoceanlog/UI/model_inference"
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

---

This deployment guide provides comprehensive instructions for deploying Nava Ocean Log in various environments. For additional support, refer to the troubleshooting section or contact the development team. 