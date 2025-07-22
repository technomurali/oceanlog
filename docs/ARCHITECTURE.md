# Nava Ocean Log - System Architecture

## Overview

Nava Ocean Log is built as a modular, scalable AI-powered container tracking system with a clear separation of concerns between the web interface, AI processing pipeline, and data management components.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Nava Ocean Log System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   Web Interface │    │  AI Processing  │    │   Output     │ │
│  │   (Streamlit)   │    │    Pipeline     │    │ Management   │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                      │      │
│           ▼                       ▼                      ▼      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   Video Upload  │    │ Container Back  │    │ Image Crops  │ │
│  │   & Validation  │    │   Detection     │    │ & Storage    │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                      │      │
│           ▼                       ▼                      ▼      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  Processing     │    │ Label Detection │    │ JSON Results │ │
│  │   Trigger       │    │   (YOLOv8)      │    │ Generation   │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                      │      │
│           ▼                       ▼                      ▼      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  Results        │    │ OCR Processing  │    │ Performance  │ │
│  │  Display        │    │  (PaddleOCR)    │    │ Metrics      │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Web Interface Layer (Streamlit)

**File**: `UI/app.py`

**Responsibilities**:
- User interaction and video upload
- Processing trigger and progress monitoring
- Results visualization and display
- Session state management

**Key Components**:
```python
# Main application structure
st.set_page_config(page_title="Nava Ocean Log", layout="wide")
st.title("Nava Ocean Log")
st.subheader("Powered by AI: Real-time Container Tracking System")
```

**Features**:
- File upload with validation (MP4, max 500MB)
- Real-time progress tracking
- Interactive results display
- Performance metrics visualization

### 2. AI Processing Pipeline

**File**: `UI/model_inference/agent_container.py`

**Responsibilities**:
- Video processing and frame extraction
- Object detection using YOLOv8 models
- Image cropping and preprocessing
- OCR text extraction

#### 2.1 Container Back Detection

**Function**: `detect_and_crop_container_backs()`

**Process Flow**:
1. **Model Loading**: Load YOLOv8 container detection model
2. **Video Processing**: Extract frames from input video
3. **Object Detection**: Detect container backs in each frame
4. **Image Cropping**: Crop detected regions with margin
5. **File Management**: Save cropped images to output directory

**Key Parameters**:
- `confidence_threshold`: 0.953 (95.3%)
- `progress_every_n`: 25 frames
- `label_of_interest`: "container_back"

#### 2.2 Label Detection

**Function**: `detect_and_crop_labels()`

**Process Flow**:
1. **Model Loading**: Load YOLOv8 label detection model
2. **Image Processing**: Process cropped container images
3. **Label Detection**: Detect various labels on container backs
4. **Region Cropping**: Extract individual label regions
5. **Organization**: Save labels to structured directory

#### 2.3 OCR Processing

**Function**: `run_paddle_ocr_to_json()`

**Process Flow**:
1. **Image Loading**: Load cropped label images
2. **Text Recognition**: Extract text using PaddleOCR
3. **Data Structuring**: Organize results by label type
4. **JSON Generation**: Create structured output file

### 3. Data Management

**Responsibilities**:
- File organization and cleanup
- Output directory management
- Performance tracking and logging

#### 3.1 Directory Structure

```
UI/
├── uploaded_video_files/     # Input video storage
├── video_outputs/           # Processing results
│   └── {video_name}_container_back/
│       ├── container_back_*.jpg    # Cropped container images
│       └── labels/                 # Detected labels
│           ├── chassis_license_plate.jpg
│           ├── container_number.jpg
│           └── ocr_results.json    # OCR results
├── models/                  # AI model storage
│   ├── container_back_identifier.pt
│   └── container_back_labeled.pt
└── model_inference/         # Processing logic
```

#### 3.2 File Cleanup

**Function**: `clean_folder_except_last_n()`

**Purpose**: Maintains disk space by keeping only the most recent processed files.

## Data Flow

### 1. Input Processing

```
Video Upload → Validation → Storage → Processing Trigger
```

**Steps**:
1. User uploads MP4 video file
2. System validates file format and size
3. Video saved to `uploaded_video_files/`
4. Processing pipeline triggered

### 2. AI Processing Pipeline

```
Video → Container Detection → Image Crops → Label Detection → OCR → JSON
```

**Detailed Flow**:
1. **Video Input**: MP4 file from upload directory
2. **Container Detection**: YOLOv8 identifies container backs
3. **Image Cropping**: Detected regions extracted with margin
4. **Label Detection**: Second YOLOv8 model finds labels
5. **OCR Processing**: PaddleOCR extracts text from labels
6. **JSON Output**: Structured results saved to file

### 3. Results Generation

```
Processing Results → Visualization → User Display
```

**Components**:
- Container detection images
- Extracted label text
- Performance metrics
- Processing timestamps

## Model Architecture

### YOLOv8 Models

#### Container Back Detection Model
- **Purpose**: Primary container identification
- **Input**: Video frames (RGB)
- **Output**: Bounding boxes for container backs
- **Confidence**: 95.3% threshold
- **Performance**: 25-30 FPS (GPU)

#### Label Detection Model
- **Purpose**: Secondary label identification
- **Input**: Cropped container images
- **Output**: Bounding boxes for labels
- **Classes**: Multiple label types
- **Performance**: 15-20 FPS (GPU)

### PaddleOCR Integration

**Configuration**:
- **Language**: English (en)
- **Model**: PP-OCRv3
- **Precision**: FP32
- **Performance**: 10-15 FPS per image

## Performance Considerations

### 1. Memory Management

**GPU Memory**:
- Model loading: ~2GB
- Batch processing: ~1GB
- Total requirement: ~4GB

**System Memory**:
- Video processing: ~2GB
- Image storage: ~1GB
- Total requirement: ~4GB

### 2. Processing Optimization

**Strategies**:
- Frame skipping for long videos
- Batch processing for multiple detections
- GPU memory cleanup between operations
- Progress tracking for user feedback

### 3. Scalability

**Horizontal Scaling**:
- Multiple processing workers
- Distributed model inference
- Load balancing for video processing

**Vertical Scaling**:
- GPU memory optimization
- Model quantization
- Batch size adjustment

## Error Handling

### 1. Video Processing Errors

**Common Issues**:
- Corrupted video files
- Unsupported codecs
- Memory limitations

**Solutions**:
- File validation before processing
- Graceful error recovery
- User-friendly error messages

### 2. Model Inference Errors

**Common Issues**:
- CUDA out of memory
- Model loading failures
- Invalid input formats

**Solutions**:
- Automatic CPU fallback
- Model verification
- Input preprocessing

### 3. OCR Processing Errors

**Common Issues**:
- Low-quality images
- Unsupported characters
- Model loading failures

**Solutions**:
- Image preprocessing
- Character encoding handling
- Alternative OCR engines

## Security Considerations

### 1. File Upload Security

- File type validation
- Size limitations
- Path traversal prevention
- Virus scanning (optional)

### 2. Model Security

- Model integrity verification
- Secure model storage
- Access control for model files

### 3. Data Privacy

- Temporary file cleanup
- User data isolation
- Secure data transmission

## Monitoring and Logging

### 1. Performance Monitoring

**Metrics Tracked**:
- Processing time per video
- Memory usage
- GPU utilization
- Error rates

### 2. System Logging

**Log Levels**:
- DEBUG: Detailed processing information
- INFO: General system status
- WARNING: Non-critical issues
- ERROR: Processing failures

### 3. User Analytics

**Tracked Events**:
- Video uploads
- Processing completions
- Error occurrences
- User interactions

## Future Enhancements

### 1. Scalability Improvements

- Microservices architecture
- Container orchestration (Docker/Kubernetes)
- Distributed processing
- Cloud deployment options

### 2. Feature Additions

- Real-time video streaming
- Multi-language OCR support
- Advanced analytics dashboard
- API endpoints for integration

### 3. Performance Optimizations

- Model quantization
- TensorRT acceleration
- Edge deployment
- Caching mechanisms

---

This architecture provides a solid foundation for the Nava Ocean Log system while maintaining flexibility for future enhancements and scalability requirements. 