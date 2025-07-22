# Contributing to Nava Ocean Log

Thank you for your interest in contributing to Nava Ocean Log! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Documentation](#documentation)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.8 or higher
- Git
- Basic knowledge of Python, computer vision, and AI/ML concepts
- Familiarity with Streamlit, PyTorch, and OpenCV (for code contributions)

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/navaoceanlog.git
   cd navaoceanlog
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-owner/navaoceanlog.git
   ```

### Development Environment

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Setup

### Project Structure

```
navaoceanlog/
├── UI/                          # Main application directory
│   ├── app.py                   # Streamlit web interface
│   ├── model_inference/         # AI processing modules
│   │   ├── agent_container.py   # Main processing pipeline
│   │   └── agent_container_video_macker.py  # Video processing
│   ├── models/                  # AI model storage
│   ├── uploaded_video_files/    # Input video storage
│   └── video_outputs/           # Processing results
├── docs/                        # Documentation
├── tests/                       # Test files
├── scripts/                     # Utility scripts
└── requirements.txt             # Dependencies
```

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Test your changes**:
   ```bash
   # Run tests
   pytest tests/
   
   # Run linting
   black UI/
   isort UI/
   flake8 UI/
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Code Style

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

#### Formatting
- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **String quotes**: Double quotes for docstrings, single quotes for strings

#### Naming Conventions
- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

#### Example
```python
"""Module docstring."""

import os
from typing import List, Optional

CONSTANT_VALUE = 42


class ContainerProcessor:
    """Class for processing container videos."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._internal_state = None
    
    def process_video(self, video_path: str) -> List[str]:
        """Process a video file and return results."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        return self._extract_results(video_path)
    
    def _extract_results(self, path: str) -> List[str]:
        """Extract results from processed video."""
        # Implementation here
        pass
```

### Type Hints

Use type hints for all function parameters and return values:

```python
from typing import List, Dict, Optional, Union

def process_container(
    video_path: str,
    confidence_threshold: float = 0.95,
    output_dir: Optional[str] = None
) -> Dict[str, Union[str, List[str]]]:
    """Process container video with type hints."""
    pass
```

### Docstrings

Follow Google-style docstrings:

```python
def detect_containers(
    model_path: str,
    video_path: str,
    confidence: float = 0.95
) -> List[Dict[str, Any]]:
    """Detect containers in video using YOLOv8 model.
    
    Args:
        model_path: Path to the YOLOv8 model file.
        video_path: Path to the input video file.
        confidence: Detection confidence threshold (0.0 to 1.0).
    
    Returns:
        List of detection results with bounding boxes and confidence scores.
    
    Raises:
        FileNotFoundError: If model or video file doesn't exist.
        ValueError: If confidence is outside valid range.
    
    Example:
        >>> results = detect_containers('model.pt', 'video.mp4', 0.9)
        >>> print(f"Found {len(results)} containers")
    """
    pass
```

### Import Organization

Organize imports in this order:

1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Standard library
import os
import sys
from pathlib import Path
from typing import List, Optional

# Third-party
import cv2
import torch
from ultralytics import YOLO

# Local
from .utils import validate_path
from .models import ContainerModel
```

## Testing

### Test Structure

Create tests in the `tests/` directory:

```
tests/
├── __init__.py
├── test_agent_container.py
├── test_video_processing.py
├── test_ocr_processing.py
└── conftest.py
```

### Writing Tests

Use pytest for testing:

```python
import pytest
from pathlib import Path
from UI.model_inference.agent_container import detect_and_crop_container_backs


class TestContainerDetection:
    """Test container detection functionality."""
    
    def test_detect_containers_valid_input(self, sample_video_path):
        """Test container detection with valid input."""
        # Arrange
        model_path = "tests/fixtures/test_model.pt"
        output_dir = "tests/output"
        
        # Act
        detect_and_crop_container_backs(
            model_path=model_path,
            video_path=sample_video_path,
            output_dir=output_dir
        )
        
        # Assert
        assert Path(output_dir).exists()
        assert len(list(Path(output_dir).glob("*.jpg"))) > 0
    
    def test_detect_containers_invalid_model(self):
        """Test container detection with invalid model path."""
        with pytest.raises(FileNotFoundError):
            detect_and_crop_container_backs(
                model_path="nonexistent_model.pt",
                video_path="test.mp4",
                output_dir="output"
            )
    
    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_confidence_thresholds(self, confidence, sample_video_path):
        """Test different confidence thresholds."""
        # Test implementation
        pass
```

### Test Fixtures

Create reusable test fixtures in `conftest.py`:

```python
import pytest
from pathlib import Path


@pytest.fixture
def sample_video_path():
    """Provide path to sample video for testing."""
    return Path("tests/fixtures/sample_video.mp4")


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide temporary output directory."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_model():
    """Provide mock YOLO model for testing."""
    # Mock implementation
    pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=UI --cov-report=html

# Run specific test file
pytest tests/test_agent_container.py

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**:
   ```bash
   pytest tests/
   ```

2. **Check code style**:
   ```bash
   black UI/
   isort UI/
   flake8 UI/
   ```

3. **Update documentation** if needed

4. **Test your changes** manually

### Pull Request Guidelines

1. **Title**: Use conventional commit format
   - `feat: add new container detection feature`
   - `fix: resolve memory leak in video processing`
   - `docs: update API documentation`

2. **Description**: Include:
   - Summary of changes
   - Motivation for changes
   - Testing performed
   - Screenshots (if UI changes)

3. **Checklist**:
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] Code follows style guidelines
   - [ ] No breaking changes (or documented)

### Example Pull Request

```markdown
## Description
Adds support for batch video processing to improve throughput for multiple video files.

## Changes
- Added `process_video_batch()` function in `agent_container.py`
- Implemented parallel processing with configurable batch size
- Added progress tracking for batch operations
- Updated documentation with batch processing examples

## Testing
- Added unit tests for batch processing
- Tested with 10 sample videos
- Verified memory usage remains stable

## Screenshots
[If applicable]

## Checklist
- [x] Tests added/updated
- [x] Documentation updated
- [x] Code follows style guidelines
- [x] No breaking changes
```

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Environment details**:
   - OS and version
   - Python version
   - Package versions
   - GPU/CPU configuration

2. **Steps to reproduce**:
   - Clear, step-by-step instructions
   - Sample data if applicable

3. **Expected vs actual behavior**:
   - What you expected to happen
   - What actually happened

4. **Error messages and logs**:
   - Full error traceback
   - Application logs

### Example Bug Report

```markdown
## Bug Description
Video processing fails with CUDA out of memory error on large files.

## Environment
- OS: Ubuntu 20.04
- Python: 3.9.7
- GPU: RTX 3080 (10GB VRAM)
- Package versions: [list relevant packages]

## Steps to Reproduce
1. Upload video file > 200MB
2. Click "Run AI Processing"
3. Error occurs after 30 seconds

## Expected Behavior
Video should process successfully with progress updates.

## Actual Behavior
Application crashes with CUDA out of memory error.

## Error Message
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

## Additional Information
- Video resolution: 1920x1080
- Video length: 5 minutes
- Memory usage before processing: 2GB
```

## Feature Requests

When requesting features:

1. **Describe the problem** you're trying to solve
2. **Explain the proposed solution**
3. **Provide use cases** and examples
4. **Consider implementation complexity**

### Example Feature Request

```markdown
## Problem
Currently, the system only supports MP4 video format, limiting compatibility with various camera systems.

## Proposed Solution
Add support for additional video formats (AVI, MOV, MKV) using FFmpeg for format conversion.

## Use Cases
- Integration with existing surveillance systems
- Support for different camera manufacturers
- Compatibility with mobile device recordings

## Implementation Considerations
- FFmpeg dependency addition
- Format detection and conversion
- Error handling for unsupported formats
```

## Documentation

### Documentation Standards

1. **Keep documentation up to date** with code changes
2. **Use clear, concise language**
3. **Include examples** and code snippets
4. **Add diagrams** for complex concepts

### Documentation Types

- **API Documentation**: Function signatures, parameters, return values
- **User Guides**: Step-by-step instructions for common tasks
- **Architecture Docs**: System design and component interactions
- **Deployment Guides**: Installation and configuration instructions

### Documentation Tools

- **Markdown**: For all documentation files
- **Sphinx**: For API documentation (optional)
- **Mermaid**: For diagrams and flowcharts

## Community Guidelines

### Code of Conduct

1. **Be respectful** and inclusive
2. **Provide constructive feedback**
3. **Help others** learn and contribute
4. **Follow project conventions**

### Communication

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

### Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

## Getting Help

### Resources

- **Documentation**: Check the `docs/` directory
- **Issues**: Search existing issues for similar problems
- **Discussions**: Ask questions in GitHub Discussions

### Contact

- **Maintainers**: [@maintainer-handles]
- **Email**: [project-email]
- **Discord/Slack**: [community-channels]

---

Thank you for contributing to Nava Ocean Log! Your contributions help make this project better for everyone. 