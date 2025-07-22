# Nava Ocean Log - API Reference

## Overview

This document provides comprehensive API documentation for the Nava Ocean Log system, including all functions, parameters, return values, and usage examples.

## Core Modules

### 1. Main Application (`UI/app.py`)

The main Streamlit application that provides the web interface for video upload, processing, and results display.

#### Configuration Constants

```python
ROOT_DIR = Path("/home/ubuntu/navaoceanlog/UI")
UPLOAD_DIR = ROOT_DIR / "uploaded_video_files"
OUTPUTS_DIR = ROOT_DIR / "video_outputs"
MODEL_INFERENCE_DIR = ROOT_DIR / "model_inference"
```

#### Session State Variables

```python
st.session_state.upload_time    # Time taken for video upload
st.session_state.exec_time      # Time taken for AI processing
```

### 2. AI Processing Pipeline (`UI/model_inference/agent_container.py`)

The core AI processing module containing all detection and OCR functions.

## Function Reference

### Container Detection Functions

#### `detect_and_crop_container_backs()`

Detects container backs in video frames and crops the detected regions.

**Signature:**
```python
def detect_and_crop_container_backs(
    model_path: str,
    video_path: str,
    output_dir: str,
    label_of_interest: str = "container_back",
    confidence_threshold: float = 0.80,
    progress_every_n: int = 50
) -> None
```

**Parameters:**
- `model_path` (str): Path to the YOLOv8 container detection model file
- `video_path` (str): Path to the input video file
- `output_dir` (str): Directory to save cropped container images
- `label_of_interest` (str, optional): Target label for detection. Default: "container_back"
- `confidence_threshold` (float, optional): Minimum confidence for detection. Default: 0.80
- `progress_every_n` (int, optional): Frequency of progress reporting. Default: 50

**Returns:** None

**Example:**
```python
detect_and_crop_container_backs(
    model_path="/path/to/container_back_identifier.pt",
    video_path="/path/to/input_video.mp4",
    output_dir="/path/to/output",
    confidence_threshold=0.953,
    progress_every_n=25
)
```

**Behavior:**
- Loads YOLOv8 model on GPU if available, otherwise CPU
- Processes video frame by frame
- Detects container backs above confidence threshold
- Crops detected regions with 10px margin
- Saves cropped images to output directory
- Provides progress updates every N frames

### Label Detection Functions

#### `detect_and_crop_labels()`

Detects labels on cropped container images and extracts individual label regions.

**Signature:**
```python
def detect_and_crop_labels(
    model_path: str,
    image_folder: str,
    output_root: str = None
) -> None
```

**Parameters:**
- `model_path` (str): Path to the YOLOv8 label detection model file
- `image_folder` (str): Directory containing cropped container images
- `output_root` (str, optional): Root directory for label outputs. Default: None (uses image_folder)

**Returns:** None

**Example:**
```python
detect_and_crop_labels(
    model_path="/path/to/container_back_labeled.pt",
    image_folder="/path/to/container_images",
    output_root="/path/to/labels"
)
```

**Behavior:**
- Loads YOLOv8 label detection model
- Processes all images in the input folder
- Detects various label types on container backs
- Creates subdirectories for each label type
- Saves individual label images to appropriate directories

### OCR Processing Functions

#### `run_paddle_ocr_to_json()`

Performs OCR on images and saves results in JSON format.

**Signature:**
```python
def run_paddle_ocr_to_json(
    image_folder: str,
    output_filename: str = "ocr_results.json",
    *,
    lang: str = "en",
    logger: logging.Logger | None = None,
    ocr_kwargs: Dict[str, Any] | None = None
) -> str
```

**Parameters:**
- `image_folder` (str): Directory containing images for OCR processing
- `output_filename` (str, optional): Name of output JSON file. Default: "ocr_results.json"
- `lang` (str, optional): OCR language. Default: "en"
- `logger` (logging.Logger, optional): Logger instance for debugging. Default: None
- `ocr_kwargs` (Dict[str, Any], optional): Additional PaddleOCR parameters. Default: None

**Returns:** str - Path to the generated JSON file

**Example:**
```python
json_path = run_paddle_ocr_to_json(
    image_folder="/path/to/label_images",
    output_filename="container_labels.json",
    lang="en"
)
```

**Output Format:**
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

#### `run_paddle_ocr_to_json_old()`

Legacy OCR function with simplified output format.

**Signature:**
```python
def run_paddle_ocr_to_json_old(
    image_folder: str,
    output_filename: str = 'ocr_results.json'
) -> None
```

**Parameters:**
- `image_folder` (str): Directory containing images for OCR
- `output_filename` (str, optional): Output JSON filename. Default: 'ocr_results.json'

**Returns:** None

### Utility Functions

#### `clean_folder_except_last_n()`

Removes all files except the last N files from a directory.

**Signature:**
```python
def clean_folder_except_last_n(
    folder_path: str,
    last_no_files: int = 1
) -> None
```

**Parameters:**
- `folder_path` (str): Directory to clean
- `last_no_files` (int, optional): Number of recent files to keep. Default: 1

**Returns:** None

**Example:**
```python
clean_folder_except_last_n("/path/to/output", last_no_files=5)
```

#### `generate_output_path()`

Generates a dynamic output path based on input video name and label.

**Signature:**
```python
def generate_output_path(
    input_path: str,
    label: str,
    output_base_path: str
) -> str
```

**Parameters:**
- `input_path` (str): Full path to input video file
- `label` (str): Label to append to output path
- `output_base_path` (str): Base output directory

**Returns:** str - Final output directory path

**Example:**
```python
output_path = generate_output_path(
    input_path="/path/to/video.mp4",
    label="container_back",
    output_base_path="/path/to/output"
)
# Returns: "/path/to/output/video_container_back"
```

### Main Processing Function

#### `run_all()`

Main entry point for the complete video processing pipeline.

**Signature:**
```python
def run_all(
    input_video_path_setup: str,
    output_dair_path: str
) -> None
```

**Parameters:**
- `input_video_path_setup` (str): Path to input video file
- `output_dair_path` (str): Base directory for output files

**Returns:** None

**Example:**
```python
run_all(
    input_video_path_setup="/path/to/container_video.mp4",
    output_dair_path="/path/to/output"
)
```

**Processing Steps:**
1. Generates output directory path
2. Runs container back detection
3. Cleans output folder (keeps last file)
4. Runs label detection on cropped images
5. Performs OCR on detected labels
6. Saves results as JSON

## Video Processing Functions (`UI/model_inference/agent_container_video_macker.py`)

### `container_back_video_gen()`

Generates a video with container back detection overlays.

**Signature:**
```python
def container_back_video_gen(video_path: str) -> str
```

**Parameters:**
- `video_path` (str): Path to input video file

**Returns:** str - Path to output video with detections

**Example:**
```python
output_video = container_back_video_gen("/path/to/input.mp4")
```

### `container_back_label_gen()`

Generates a video with label detection overlays.

**Signature:**
```python
def container_back_label_gen(video_path: str) -> str
```

**Parameters:**
- `video_path` (str): Path to input video file

**Returns:** str - Path to output video with label detections

### `generate_final_output()`

Combines both container and label detection in a single pipeline.

**Signature:**
```python
def generate_final_output(input_video_path: str) -> str
```

**Parameters:**
- `input_video_path` (str): Path to input video file

**Returns:** str - Path to final output video

## Error Handling

### Common Exceptions

#### `IOError`
Raised when video file cannot be opened or read.

**Handling:**
```python
try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
except IOError as e:
    print(f"Video processing error: {e}")
```

#### `RuntimeError`
Raised when CUDA operations fail (out of memory, etc.).

**Handling:**
```python
try:
    model = YOLO(model_path).to(device)
except RuntimeError as e:
    print(f"GPU error: {e}")
    # Fallback to CPU
    device = "cpu"
    model = YOLO(model_path).to(device)
```

#### `ImportError`
Raised when required dependencies are not installed.

**Handling:**
```python
try:
    from paddleocr import PaddleOCR
except ImportError as exc:
    raise ImportError(
        "PaddleOCR is not installed. Install with: pip install paddleocr"
    ) from exc
```

## Configuration

### Model Paths

```python
# Default model paths
CONTAINER_BACK_MODEL = "/home/ubuntu/navaoceanlog/UI/models/container_back_identifier.pt"
CONTAINER_LABEL_MODEL = "/home/ubuntu/navaoceanlog/UI/models/container_back_labeled.pt"
```

### Processing Parameters

```python
# Default confidence thresholds
CONTAINER_CONFIDENCE = 0.953  # 95.3%
LABEL_CONFIDENCE = 0.5        # 50%

# Progress reporting
PROGRESS_FREQUENCY = 25       # Report every 25 frames

# File management
KEEP_LAST_FILES = 1          # Keep only the most recent file
```

### OCR Configuration

```python
# PaddleOCR settings
OCR_LANGUAGE = "en"
OCR_MODEL = "PP-OCRv3"
OCR_PRECISION = "FP32"

# Output settings
DEFAULT_JSON_FILENAME = "ocr_results.json"
```

## Performance Optimization

### GPU Acceleration

```python
# Check GPU availability
use_cuda = torch.cuda.is_available()
device_str = "cuda" if use_cuda else "cpu"

# Load model on appropriate device
model = YOLO(model_path).to(device_str)
```

### Memory Management

```python
# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Batch processing
for batch in batches:
    results = model.predict(batch)
    torch.cuda.empty_cache()  # Clear after each batch
```

### Progress Tracking

```python
# Custom progress callback
def progress_callback(current, total):
    progress = (current / total) * 100
    print(f"Processing: {progress:.1f}%")

# Use in processing loop
for i, frame in enumerate(frames):
    if i % progress_every_n == 0:
        progress_callback(i, len(frames))
```

## Integration Examples

### Basic Usage

```python
from model_inference.agent_container import run_all

# Simple video processing
run_all("/path/to/video.mp4", "/path/to/output")
```

### Advanced Usage

```python
from model_inference.agent_container import (
    detect_and_crop_container_backs,
    detect_and_crop_labels,
    run_paddle_ocr_to_json
)

# Step-by-step processing
# 1. Detect containers
detect_and_crop_container_backs(
    model_path="/path/to/container_model.pt",
    video_path="/path/to/video.mp4",
    output_dir="/path/to/containers",
    confidence_threshold=0.95
)

# 2. Detect labels
detect_and_crop_labels(
    model_path="/path/to/label_model.pt",
    image_folder="/path/to/containers"
)

# 3. Extract text
json_path = run_paddle_ocr_to_json("/path/to/containers/labels")
```

### Batch Processing

```python
import os
from pathlib import Path

def process_video_batch(video_dir, output_dir):
    """Process all videos in a directory."""
    video_files = Path(video_dir).glob("*.mp4")
    
    for video_file in video_files:
        print(f"Processing: {video_file.name}")
        try:
            run_all(str(video_file), output_dir)
            print(f"Completed: {video_file.name}")
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")

# Usage
process_video_batch("/path/to/videos", "/path/to/outputs")
```

### Custom OCR Processing

```python
def custom_ocr_processing(image_folder, output_file):
    """Custom OCR with specific parameters."""
    ocr_kwargs = {
        "use_angle_cls": True,
        "lang": "en",
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.5
    }
    
    return run_paddle_ocr_to_json(
        image_folder=image_folder,
        output_filename=output_file,
        ocr_kwargs=ocr_kwargs
    )
```

## Testing

### Unit Tests

```python
import unittest
from model_inference.agent_container import generate_output_path

class TestAgentContainer(unittest.TestCase):
    def test_generate_output_path(self):
        result = generate_output_path(
            "/path/to/video.mp4",
            "container_back",
            "/output"
        )
        self.assertEqual(result, "/output/video_container_back")

if __name__ == "__main__":
    unittest.main()
```

### Integration Tests

```python
def test_full_pipeline():
    """Test complete processing pipeline."""
    test_video = "/path/to/test_video.mp4"
    test_output = "/path/to/test_output"
    
    try:
        run_all(test_video, test_output)
        
        # Verify outputs exist
        assert Path(test_output).exists()
        assert Path(f"{test_output}/labels/ocr_results.json").exists()
        
        print("Pipeline test passed!")
    except Exception as e:
        print(f"Pipeline test failed: {e}")
```

---

This API reference provides comprehensive documentation for all functions and components of the Nava Ocean Log system. For additional examples and use cases, refer to the main README and architecture documentation. 