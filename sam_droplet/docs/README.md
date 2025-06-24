# SAM Droplet Segmentation - Documentation

## 📖 Overview

Welcome to the SAM Droplet Segmentation application documentation. This application provides an advanced image segmentation solution using Meta's Segment Anything Model (SAM) with specialized filtering capabilities for droplet analysis.

### Key Features

- 🔬 **AI-Powered Segmentation**: Uses Mobile SAM for efficient, accurate segmentation
- 🎯 **Pixel-Level Filtering**: Advanced filtering based on intensity, texture, and spatial properties
- 🌐 **Web Interface**: User-friendly web interface for interactive segmentation
- 📡 **REST API**: Comprehensive API for programmatic access
- ⚡ **GPU Acceleration**: CUDA support for fast processing
- 🔄 **Interactive Refinement**: Click-to-filter false positives
- 📊 **Statistical Analysis**: Detailed pixel and spatial statistics for each mask

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB+ recommended)
- Optional: NVIDIA GPU with CUDA support

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd sam_droplet

# Install dependencies
pip install -r requirements_server.txt

# Download SAM model and place in model/ directory
# Start the server
python app.py
```

Navigate to `http://localhost:9487` to access the web interface.

## 📚 Documentation Structure

This documentation is organized into several focused guides:

### 📋 [API Reference](api_reference.md)
Complete REST API documentation including:
- All endpoints with request/response formats
- Filter parameters and data types
- Error handling and status codes
- Client SDK examples in Python and JavaScript
- Best practices for API usage

### 🔧 [Implementation Guide](implementation_guide.md)
Technical deep-dive covering:
- System architecture and design principles
- Core components and their interactions
- Model integration and configuration
- Filtering system architecture
- Code examples for extending functionality

### 🚀 [Deployment Guide](deployment_guide.md)
Comprehensive deployment instructions:
- Development and production deployment options
- Docker containerization
- Cloud deployment (AWS, GCP, Azure)
- Performance tuning and optimization
- Security considerations

### 🎯 [Filter Guide](../README_FILTERS.md)
Specialized guide for the filtering system:
- Available filter types and use cases
- Interactive filtering features
- Preset filters for common scenarios
- Advanced filtering techniques

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Interface                          │
│                   (HTML/CSS/JS)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Flask API                               │
│                   (app.py)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │   Health    │  │ Segmentation│  │    Analysis     │    │
│  │   Check     │  │ Endpoints   │  │   Endpoints     │    │
│  └─────────────┘  └─────────────┘  └─────────────────┘    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                Core Processing                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │  SAM Model  │  │ Filtering   │  │  Image Utils    │    │
│  │ (mobile_sam)│  │ (filters.py)│  │   (OpenCV)      │    │
│  └─────────────┘  └─────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Use Cases

### Research Applications
- **Droplet Analysis**: Segmentation and measurement of droplets in microfluidics
- **Cell Biology**: Analysis of cellular structures and organelles
- **Material Science**: Particle size distribution analysis
- **Quality Control**: Automated defect detection in manufacturing

### Filtering Scenarios
- **Bright Droplets**: Isolate bright objects against dark backgrounds
- **Size Filtering**: Remove noise or focus on specific size ranges
- **Edge Exclusion**: Analyze only complete objects (not cut off by image edges)
- **Texture Analysis**: Distinguish between smooth and textured objects

## 📊 Example Workflows

### Basic Segmentation
1. Upload image via web interface or API
2. Generate masks using SAM
3. Review results with interactive overlay
4. Download masks or use API response

### Filtered Analysis
1. Upload and analyze image to understand pixel distribution
2. Set appropriate filters based on analysis
3. Generate filtered masks
4. Refine results with interactive click filtering
5. Export final results

### Batch Processing
1. Use Python SDK for automated processing
2. Apply consistent filters across image sets
3. Collect statistics and export data
4. Generate reports and visualizations

## 🔍 Filter Types Quick Reference

| Filter Type | Purpose | Range | Example |
|-------------|---------|-------|---------|
| `mean_min/max` | Brightness filtering | 0-255 | Bright droplets: `{"mean_min": 150}` |
| `area_min/max` | Size filtering | 0+ pixels | Remove noise: `{"area_min": 50}` |
| `std_min/max` | Texture filtering | 0+ | Uniform objects: `{"std_max": 15}` |
| `min_edge_distance` | Edge proximity | 0+ pixels | Complete objects: `{"min_edge_distance": 20}` |
| `exclude_edge_touching` | Edge exclusion | true/false | No edge contact: `{"exclude_edge_touching": true}` |

## 🛠️ API Quick Reference

### Health Check
```bash
curl http://localhost:9487/health
```

### Segment Image
```bash
curl -X POST http://localhost:9487/segment_file \
  -F "file=@image.jpg" \
  -F 'filters={"mean_min": 140, "area_min": 50}'
```

### Analyze Image
```bash
curl -X POST http://localhost:9487/analyze_image \
  -F "file=@image.jpg"
```

## 🔧 Configuration

### Environment Variables
```env
# Server configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=9487
FLASK_DEBUG=False

# Model configuration  
SAM_MODEL_PATH=model/mobile_sam.pt

# Performance tuning
OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0
```

### Model Requirements
- **Mobile SAM**: `mobile_sam.pt` in `model/` directory
- **Download**: Available from [Mobile SAM GitHub](https://github.com/ChaoningZhang/MobileSAM)
- **Size**: ~39MB (much smaller than full SAM)
- **Performance**: Good balance of speed and accuracy

## 📈 Performance Guidelines

### Hardware Recommendations
- **CPU**: 4+ cores, 8+ recommended
- **RAM**: 8GB minimum, 16GB+ recommended  
- **GPU**: NVIDIA RTX series with 6GB+ VRAM (optional but recommended)
- **Storage**: SSD recommended for model loading

### Optimization Tips
- Use GPU when available for 5-10x speedup
- Resize large images (>2048px) before processing
- Apply filters to reduce post-processing overhead
- Use appropriate server configuration for your traffic

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**: Ensure `mobile_sam.pt` is in `model/` directory
2. **GPU errors**: Check CUDA installation and available memory
3. **Slow processing**: Consider image size reduction or CPU optimization
4. **Memory errors**: Increase system RAM or reduce batch size

### Getting Help

1. Check the specific guides for detailed troubleshooting
2. Review application logs for error details
3. Test with smaller images to isolate issues
4. Verify system requirements are met

## 📝 Development

### Project Structure
```
sam_droplet/
├── app.py                    # Main Flask application
├── src/sam_droplet/
│   ├── filters.py           # Filtering engine
│   └── ...
├── static/                  # Web interface files
├── model/                   # SAM model files
├── docs/                    # Documentation
├── tests/                   # Test suite
└── requirements*.txt        # Dependencies
```

### Contributing
1. Read the [Implementation Guide](implementation_guide.md) for technical details
2. Follow the existing code style and patterns
3. Add tests for new functionality
4. Update documentation as needed

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- **Meta AI** for the Segment Anything Model
- **Mobile SAM** team for the efficient implementation
- **OpenCV** and **PyTorch** communities for excellent libraries
- **Flask** ecosystem for web framework support

---

**Need help?** Check the specific documentation guides linked above, or review the inline code comments for implementation details. 