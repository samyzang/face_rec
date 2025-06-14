# Facial Recognition System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready facial recognition system built with Python, featuring comprehensive error handling, caching, logging, and professional-grade architecture.

## 🚀 Features

- **High-Performance Recognition**: Utilizes state-of-the-art face_recognition library
- **Intelligent Caching**: Automatic encoding caching with staleness detection
- **Comprehensive Logging**: Full audit trail with configurable log levels
- **Error Resilience**: Robust error handling and graceful degradation
- **Flexible Configuration**: Customizable confidence thresholds and models
- **Professional Architecture**: Clean, maintainable, and extensible codebase
- **Command-Line Interface**: Easy-to-use CLI for batch processing
- **Multiple Image Formats**: Support for JPG, PNG, BMP, TIFF formats
- **Face Management**: Add/remove faces dynamically
- **Detailed Reporting**: Confidence scores and system statistics

## 📋 Requirements

- Python 3.8 or higher
- OpenCV 4.x
- face_recognition library
- NumPy
- Pillow

## 🛠️ Installation

### Using pip (Recommended)

```bash
# Install required packages
pip install face-recognition opencv-python numpy pillow

# Clone the repository
git clone https://github.com/RavenConsulting/facial-recognition.git
cd facial-recognition
```

### Using conda

```bash
# Create a new environment
conda create -n face_rec python=3.9
conda activate face_rec

# Install packages
conda install -c conda-forge dlib
pip install face-recognition opencv-python numpy pillow
```

### Docker Setup

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "face_recognition_system.py", "--help"]
```

## 🏗️ Project Structure

```
facial-recognition/
├── face_recognition_system.py    # Main system implementation
├── faces/                        # Directory for known faces
│   ├── john_doe.jpg
│   ├── jane_smith.png
│   └── ...
├── tests/                        # Test images
│   └── test_group.jpg
├── logs/                         # Log files
├── requirements.txt              # Python dependencies
├── config.json                   # Configuration file
├── face_encodings.pkl           # Cached encodings (auto-generated)
└── README.md
```

## 🚦 Quick Start

### 1. Prepare Known Faces

Add images of known individuals to the `faces/` directory:

```bash
mkdir faces
# Add images named as: person_name.jpg
cp john_doe.jpg faces/
cp jane_smith.png faces/
```

### 2. Basic Usage

```python
from face_recognition_system import FaceRecognitionSystem

# Initialize the system
face_system = FaceRecognitionSystem(
    faces_directory="./faces",
    confidence_threshold=0.6
)

# Recognize faces in an image
matches = face_system.recognize_faces("test_image.jpg")

# Display results
for match in matches:
    print(f"{match.name}: {match.confidence:.3f} ({'Known' if match.is_known else 'Unknown'})")

# Save annotated image
face_system.draw_results("test_image.jpg", matches, "output.jpg")
```

### 3. Command Line Usage

```bash
# Basic face recognition
python face_recognition_system.py --image test.jpg

# Custom settings
python face_recognition_system.py \
    --image group_photo.jpg \
    --faces-dir ./company_faces \
    --confidence 0.7 \
    --model cnn \
    --output annotated_result.jpg

# System information
python face_recognition_system.py --info
```

## 🔧 Configuration

### System Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `confidence_threshold` | Minimum confidence for positive identification | 0.6 | 0.0-1.0 |
| `tolerance` | Face comparison tolerance (lower = stricter) | 0.6 | 0.0-1.0 |
| `model` | Face detection model | 'hog' | 'hog', 'cnn' |

### Configuration File Example

```json
{
    "face_recognition": {
        "faces_directory": "./faces",
        "cache_file": "./face_encodings.pkl",
        "confidence_threshold": 0.6,
        "tolerance": 0.6,
        "model": "hog",
        "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    },
    "logging": {
        "level": "INFO",
        "file": "face_recognition.log",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}
```

## 📊 Performance Optimization

### Model Selection

- **HOG Model**: Faster, CPU-optimized, good for real-time applications
- **CNN Model**: More accurate, GPU-accelerated, better for batch processing

### Memory Management

- Automatic encoding caching reduces startup time
- Lazy loading of face encodings
- Efficient memory usage with numpy arrays

### Scalability Tips

```python
# For large datasets, consider batch processing
def process_large_dataset(image_paths, batch_size=50):
    face_system = FaceRecognitionSystem()
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        for image_path in batch:
            matches = face_system.recognize_faces(image_path)
            # Process matches...
```

## 🧪 Testing

### Unit Tests

```bash
# Run basic tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=face_recognition_system
```

### Performance Benchmarking

```python
import time
from face_recognition_system import FaceRecognitionSystem

# Benchmark recognition speed
face_system = FaceRecognitionSystem()

start_time = time.time()
matches = face_system.recognize_faces("test_image.jpg")
processing_time = time.time() - start_time

print(f"Processing time: {processing_time:.3f} seconds")
print(f"Faces per second: {len(matches) / processing_time:.2f}")
```

## 🔍 API Reference

### FaceRecognitionSystem Class

#### Initialization
```python
FaceRecognitionSystem(
    faces_directory: str = "./faces",
    cache_file: str = "./face_encodings.pkl",
    confidence_threshold: float = 0.6,
    model: str = "hog",
    tolerance: float = 0.6
)
```

#### Key Methods

- `recognize_faces(image_path: str) -> List[FaceMatch]`
- `add_face(image_path: str, person_name: str) -> bool`
- `remove_face(person_name: str) -> bool`
- `draw_results(image_path: str, matches: List[FaceMatch], output_path: str) -> str`
- `get_system_info() -> Dict`

### FaceMatch Dataclass

```python
@dataclass
class FaceMatch:
    name: str                                    # Person's name or "Unknown"
    confidence: float                            # Recognition confidence (0.0-1.0)
    bounding_box: Tuple[int, int, int, int]     # Face location (top, right, bottom, left)
    is_known: bool                              # Whether face is recognized
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'face_recognition'`
```bash
# Solution: Install dlib first (required dependency)
pip install cmake
pip install dlib
pip install face-recognition
```

**Issue**: `No faces detected in image`
```python
# Solution: Check image quality and try different model
face_system = FaceRecognitionSystem(model="cnn")  # More sensitive
```

**Issue**: `Poor recognition accuracy`
```python
# Solution: Adjust tolerance and confidence thresholds
face_system = FaceRecognitionSystem(
    tolerance=0.5,          # Stricter matching
    confidence_threshold=0.7  # Higher confidence requirement
)
```

### Debug Mode

```python
import logging
logging.getLogger('face_recognition_system').setLevel(logging.DEBUG)

# Enable detailed logging
face_system = FaceRecognitionSystem()
```

## 📈 Use Cases

### Security Systems
- Access control and authentication
- Surveillance and monitoring
- Visitor management

### Business Applications
- Employee attendance tracking
- Customer recognition systems
- Event management and check-in

### Integration Examples

```python
# Flask web service
from flask import Flask, request, jsonify
from face_recognition_system import FaceRecognitionSystem

app = Flask(__name__)
face_system = FaceRecognitionSystem()

@app.route('/recognize', methods=['POST'])
def recognize_face():
    file = request.files['image']
    file.save('temp_image.jpg')
    
    matches = face_system.recognize_faces('temp_image.jpg')
    
    return jsonify([{
        'name': match.name,
        'confidence': match.confidence,
        'is_known': match.is_known
    } for match in matches])
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow PEP 8 style guidelines
4. Add comprehensive tests
5. Update documentation
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/RavenConsulting/facial-recognition.git
cd facial-recognition

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [face_recognition library](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [dlib library](http://dlib.net/) for machine learning algorithms
- OpenCV community for computer vision tools

## 📞 Support

For support and questions:

- 📧 Email: support@ravenconsulting.com
- 🐛 Issues: [GitHub Issues](https://github.com/RavenConsulting/facial-recognition/issues)
- 📚 Documentation: [Wiki](https://github.com/RavenConsulting/facial-recognition/wiki)

## 🔄 Changelog

### Version 2.0.0 (Current)
- Complete rewrite with professional architecture
- Added comprehensive error handling and logging
- Implemented intelligent caching system
- Enhanced CLI interface with advanced options
- Added face management capabilities (add/remove)
- Improved performance with optimized algorithms
- Added detailed documentation and examples
- Implemented configuration management
- Added comprehensive test suite
- Enhanced security and validation

### Version 1.0.0
- Basic facial recognition functionality
- Simple face encoding and matching
- Basic OpenCV integration

---

*Built with ❤️ by Raven Consulting*