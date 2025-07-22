# Nava Ocean Log - Quick Start Guide

Get up and running with Nava Ocean Log in minutes! This guide will help you set up and use the AI-powered container tracking system.

## 🚀 Quick Setup (5 minutes)

### Prerequisites Check

Before starting, ensure you have:

- **Python 3.8+** installed
- **8GB+ RAM** available
- **20GB+ free disk space**
- **GPU** (optional, but recommended for better performance)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd navaoceanlog

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Create models directory
mkdir -p UI/models

# Download pre-trained models (replace with actual URLs)
wget -O UI/models/container_back_identifier.pt <model-url-1>
wget -O UI/models/container_back_labeled.pt <model-url-2>
```

### 3. Run the Application

```bash
# Navigate to UI directory
cd UI

# Start the application
streamlit run app.py --server.fileWatcherType none
```

### 4. Access the Web Interface

Open your browser and go to: `http://localhost:8501`

## 🎯 First Use

### Upload a Video

1. **Prepare your video**: Ensure it's in MP4 format and under 500MB
2. **Upload**: Use the file uploader in the web interface
3. **Wait**: The system will save your video to the server

### Process the Video

1. **Click "Run AI Processing"**: This starts the AI analysis
2. **Monitor Progress**: Watch the progress bar and status updates
3. **Wait for Completion**: Processing typically takes 30-120 seconds

### View Results

1. **Container Image**: See the detected container back
2. **Extracted Text**: View text extracted from container labels
3. **Performance Metrics**: Check processing times and efficiency

## 📊 Understanding Results

### Container Detection Results

The system provides:

- **Container Back Image**: Cropped image of the detected container
- **Detection Confidence**: How certain the AI is about the detection
- **Processing Time**: How long the analysis took

### OCR Results

Text extraction results include:

- **Label Names**: Types of labels detected (e.g., "chassis_license_plate")
- **Extracted Text**: Actual text found on each label
- **JSON Output**: Structured data for further processing

### Example Output

```json
[
  {
    "labelname": "chassis_license_plate",
    "text": ["ABC123", "XYZ789"]
  },
  {
    "labelname": "container_number",
    "text": ["ABCD1234567"]
  }
]
```

## ⚡ Performance Tips

### For Better Results

1. **Video Quality**: Use high-resolution videos (1080p or higher)
2. **Lighting**: Ensure good lighting conditions
3. **Camera Angle**: Position camera to capture full container backs
4. **Stability**: Use stable camera mounting to reduce motion blur

### For Faster Processing

1. **GPU Acceleration**: Use CUDA-compatible GPU if available
2. **Video Length**: Shorter videos process faster
3. **Resolution**: Lower resolution videos process faster
4. **System Resources**: Close other applications during processing

## 🔧 Common Issues & Solutions

### "CUDA out of memory" Error

**Problem**: GPU memory insufficient for processing

**Solutions**:
- Reduce video resolution
- Use CPU processing instead
- Process shorter video segments
- Close other GPU-intensive applications

### "Model not found" Error

**Problem**: AI models not downloaded or corrupted

**Solutions**:
- Verify models are in `UI/models/` directory
- Re-download model files
- Check file permissions

### "Video cannot be opened" Error

**Problem**: Unsupported video format or corrupted file

**Solutions**:
- Convert video to MP4 format
- Check video file integrity
- Ensure video codec is supported

### Slow Processing

**Problem**: Processing taking too long

**Solutions**:
- Check if GPU is being used
- Reduce video resolution
- Close other applications
- Use shorter video clips

## 📁 File Organization

After processing, your files are organized as:

```
UI/
├── uploaded_video_files/          # Your uploaded videos
├── video_outputs/                 # Processing results
│   └── {video_name}_container_back/
│       ├── container_back_*.jpg   # Detected containers
│       └── labels/                # Extracted labels
│           ├── *.jpg              # Individual label images
│           └── ocr_results.json   # Text extraction results
└── models/                        # AI models
```

## 🎛️ Configuration Options

### Adjusting Detection Sensitivity

You can modify confidence thresholds in the code:

```python
# In agent_container.py
confidence_threshold=0.953  # 95.3% confidence
```

### Changing Processing Parameters

```python
# Progress reporting frequency
progress_every_n=25  # Report every 25 frames

# File cleanup
last_no_files=1  # Keep only the most recent file
```

## 🔄 Batch Processing

For multiple videos, you can use the command line:

```bash
# Process a single video
python UI/model_inference/agent_container.py

# Or use the main function
python -c "
from UI.model_inference.agent_container import run_all
run_all('/path/to/video.mp4', '/path/to/output')
"
```

## 📈 Monitoring Performance

### Check System Resources

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor system resources
htop

# Check disk space
df -h
```

### Performance Metrics

The web interface shows:
- **Upload Time**: Time to save video to server
- **Processing Time**: AI analysis duration
- **Total Time**: Complete operation time

## 🆘 Getting Help

### Documentation

- **Full Documentation**: See `docs/` directory
- **API Reference**: `docs/API_REFERENCE.md`
- **Architecture**: `docs/ARCHITECTURE.md`

### Troubleshooting

- **Common Issues**: Check this guide's troubleshooting section
- **Error Messages**: Search documentation for specific errors
- **Performance**: Review performance optimization tips

### Support

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides and examples
- **Community**: Connect with other users

## 🎉 Next Steps

Now that you're up and running:

1. **Experiment**: Try different video types and settings
2. **Optimize**: Adjust parameters for your specific use case
3. **Integrate**: Use the JSON output in your applications
4. **Contribute**: Help improve the project (see `docs/CONTRIBUTING.md`)

## 📋 Quick Reference

### Essential Commands

```bash
# Start application
cd UI && streamlit run app.py --server.fileWatcherType none

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Test model loading
python -c "from ultralytics import YOLO; model = YOLO('UI/models/container_back_identifier.pt')"

# Process video programmatically
python -c "from UI.model_inference.agent_container import run_all; run_all('video.mp4', 'output')"
```

### Key Files

- `UI/app.py`: Main web application
- `UI/model_inference/agent_container.py`: AI processing pipeline
- `UI/models/`: AI model storage
- `requirements.txt`: Python dependencies

### Important URLs

- **Local Application**: `http://localhost:8501`
- **Documentation**: `docs/` directory
- **GitHub Repository**: `<repository-url>`

---

**Congratulations!** You're now ready to use Nava Ocean Log for AI-powered container tracking. For advanced usage and customization, refer to the full documentation in the `docs/` directory. 