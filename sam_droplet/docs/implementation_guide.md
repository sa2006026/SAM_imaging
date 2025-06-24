# SAM Droplet Segmentation - Implementation Guide

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Model Integration](#model-integration)
4. [Filtering System](#filtering-system)
5. [Web Interface](#web-interface)
6. [Deployment](#deployment)
7. [Extending the System](#extending-the-system)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)

## Architecture Overview

The SAM Droplet Segmentation application follows a modular architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │───▶│  Flask API      │───▶│  SAM Model      │
│   (HTML/JS/CSS) │    │  (app.py)       │    │  (mobile_sam)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ Filtering Engine│
                       │ (filters.py)    │
                       └─────────────────┘
```

### Key Design Principles

1. **Separation of Concerns**: Model loading, image processing, and filtering are separate modules
2. **Stateless API**: Each request is independent, no session state maintained
3. **Lazy Loading**: SAM model is loaded on first use, not startup
4. **Cross-Platform**: Works on Windows, Linux, and macOS
5. **Extensible**: Easy to add new filters and endpoints

## Core Components

### 1. Flask Application (`app.py`)

The main application file containing:

```python
# Global model variables
sam_model = None
mask_generator = None

# Core endpoints
@app.route('/health')           # Health check
@app.route('/segment_file')     # File upload segmentation  
@app.route('/segment')          # Base64 segmentation
@app.route('/analyze_image')    # Image analysis
```

#### Key Functions

- **`initialize_sam()`**: Loads the SAM model on first use
- **`process_image_from_base64()`**: Converts base64 images to numpy arrays
- **`masks_to_base64_images()`**: Converts SAM masks to base64 PNGs
- **`convert_numpy_types()`**: Handles JSON serialization of numpy types

### 2. Filtering Engine (`src/sam_droplet/filters.py`)

Provides pixel-level analysis and filtering:

```python
# Core analysis functions
def analyze_mask_pixels(image, mask) -> Dict[str, float]
def analyze_edge_proximity(mask, image_shape) -> Dict[str, Any] 
def analyze_image_statistics(image) -> Dict[str, Any]

# Filtering function
def filter_masks_by_criteria(masks, image, filters) -> List[Dict]

# Preset filters
class FilterPresets:
    @staticmethod
    def bright_droplets(image_stats) -> Dict[str, float]
    # ... other presets
```

### 3. Web Interface (`static/`)

Frontend components:
- **HTML**: Single-page interface for image upload and visualization
- **CSS**: Responsive styling with modern design
- **JavaScript**: Handles file upload, API calls, and mask display

## Model Integration

### SAM Model Loading

The application uses Mobile SAM for efficiency:

```python
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

def initialize_sam():
    global sam_model, mask_generator
    
    model_type = "vit_t"  # Vision Transformer Tiny
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam_model = sam_model_registry[model_type](checkpoint=model_path)
    sam_model.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(sam_model)
```

### Model Configuration

Key parameters for `SamAutomaticMaskGenerator`:

```python
mask_generator = SamAutomaticMaskGenerator(
    model=sam_model,
    points_per_side=32,           # Grid points for segmentation
    pred_iou_thresh=0.88,         # IoU threshold for predictions
    stability_score_thresh=0.95,  # Stability threshold
    crop_n_layers=1,              # Number of crop layers
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100      # Minimum mask area
)
```

### Model Files

Expected model location:
```
sam_droplet/
├── model/
│   └── mobile_sam.pt  # Mobile SAM checkpoint
└── app.py
```

Download from: [Mobile SAM GitHub](https://github.com/ChaoningZhang/MobileSAM)

## Filtering System

### Filter Architecture

The filtering system processes masks in two stages:

1. **Analysis Stage**: Extract pixel and spatial statistics
2. **Filtering Stage**: Apply criteria to include/exclude masks

```python
# Analysis stage
for mask in masks:
    mask['pixel_stats'] = analyze_mask_pixels(image, mask['segmentation'])
    mask['edge_stats'] = analyze_edge_proximity(mask['segmentation'], image.shape)

# Filtering stage  
filtered_masks = filter_masks_by_criteria(masks, image, filters)
```

### Available Filter Types

#### Pixel Intensity Filters
- `mean_min/max`: Average pixel brightness
- `min_threshold/max_threshold`: Extreme pixel values
- `median_min/max`: Median pixel brightness
- `std_min/max`: Pixel intensity variation (texture)

#### Spatial Filters
- `area_min/max`: Object size in pixels
- `min_edge_distance`: Distance from image edges
- `exclude_edge_touching`: Remove edge-touching objects

### Adding Custom Filters

To add a new filter type:

1. **Add analysis function** (if needed):
```python
def analyze_custom_property(image, mask):
    """Analyze custom property of mask."""
    # Your analysis code here
    return custom_value
```

2. **Update filtering logic**:
```python
def filter_masks_by_criteria(masks, image, filters):
    # ... existing code ...
    
    # Custom filter
    if 'custom_filter' in filters:
        custom_value = analyze_custom_property(image, mask['segmentation'])
        if custom_value < filters['custom_filter']:
            should_include = False
    
    # ... rest of function ...
```

3. **Add to API documentation** and update frontend if needed.

### Filter Presets

The system includes preset filters for common use cases:

```python
class FilterPresets:
    @staticmethod
    def bright_droplets(image_stats):
        """Filter for bright, medium-sized droplets."""
        mean_intensity = image_stats['overall_stats']['mean_intensity']
        return {
            'mean_min': mean_intensity + 40,
            'area_min': 50,
            'area_max': 5000,
            'std_max': 20
        }
```

## Web Interface

### Frontend Architecture

The web interface is a single-page application with:

```html
<!-- Main components -->
<div id="upload-area">       <!-- File upload zone -->
<div id="filter-panel">      <!-- Filter controls -->  
<div id="results-section">   <!-- Results display -->
<div id="mask-overlay">      <!-- Interactive overlay -->
```

### Key JavaScript Functions

```javascript
// Core functions
async function uploadAndSegment(file, filters)
async function analyzeImage(file) 
function displayResults(data)
function setupMaskOverlay(masks)

// Interactive features
function handleMaskClick(maskId)
function updateMaskVisibility()
function resetFilters()
```

### Styling

The interface uses:
- **CSS Grid/Flexbox**: Responsive layout
- **CSS Variables**: Consistent theming
- **Hover Effects**: Interactive feedback
- **Loading States**: Progress indicators

## Deployment

### Development Deployment

For development and testing:

```bash
cd sam_droplet
python app.py
# Server runs on http://localhost:9487
```

### Production Deployment

#### Option 1: Waitress (Windows/Cross-platform)

```python
# waitress_config.py
from waitress import serve
from app import app

if __name__ == '__main__':
    serve(app, 
          host='0.0.0.0', 
          port=5000,
          threads=4,
          connection_limit=100,
          channel_timeout=300)
```

#### Option 2: Gunicorn (Linux/macOS)

```bash
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 300 app:app
```

#### Option 3: Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_server.txt .
RUN pip install -r requirements_server.txt

COPY . .
EXPOSE 5000

CMD ["python", "waitress_config.py"]
```

### Environment Configuration

Key environment variables:

```bash
# Model configuration
SAM_MODEL_PATH=/path/to/model/mobile_sam.pt
CUDA_VISIBLE_DEVICES=0

# Server configuration  
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False

# Performance tuning
OMP_NUM_THREADS=4
MALLOC_TRIM_THRESHOLD_=100000
```

## Extending the System

### Adding New Endpoints

To add a new API endpoint:

```python
@app.route('/new_endpoint', methods=['POST'])
def new_endpoint():
    try:
        # Validate input
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Process request
        result = process_new_functionality(data)
        
        # Return response
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Adding New Model Types

To support different SAM variants:

```python
def initialize_sam(model_type='mobile_sam'):
    if model_type == 'mobile_sam':
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
        model_key = "vit_t"
    elif model_type == 'sam_hq':
        from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator  
        model_key = "vit_h"
    # ... other models
```

### Custom Processing Pipelines

To add preprocessing or postprocessing:

```python
def segment_image_with_pipeline():
    # Preprocessing
    image = preprocess_image(image_array)
    
    # Segmentation
    masks = mask_generator.generate(image)
    
    # Postprocessing
    masks = postprocess_masks(masks, image)
    
    # Filtering
    if filters:
        masks = filter_masks_by_criteria(masks, image, filters)
    
    return masks
```

## Performance Optimization

### GPU Optimization

```python
# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
# Clear GPU cache between requests
torch.cuda.empty_cache()

# Use mixed precision for faster inference
with torch.cuda.amp.autocast():
    masks = mask_generator.generate(image)
```

### Memory Management

```python
# Process large images in chunks
def process_large_image(image, max_size=2048):
    if max(image.shape[:2]) > max_size:
        # Resize image
        scale = max_size / max(image.shape[:2])
        new_shape = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_shape)
    
    return image

# Batch processing for multiple images
def process_image_batch(images, batch_size=4):
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = [mask_generator.generate(img) for img in batch]
        results.extend(batch_results)
    return results
```

### Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=128)
def generate_masks_cached(image_hash):
    """Cache mask generation results."""
    # Note: This is a simplified example
    # In practice, you'd need to serialize/deserialize properly
    return mask_generator.generate(image)

def get_image_hash(image_array):
    """Generate hash for image caching."""
    return hashlib.md5(image_array.tobytes()).hexdigest()
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors

```python
# Check model file exists
model_path = Path("model/mobile_sam.pt")
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

# Check model compatibility
try:
    sam_model = sam_model_registry["vit_t"](checkpoint=str(model_path))
except Exception as e:
    print(f"Model loading error: {e}")
```

#### 2. GPU Memory Issues

```bash
# Monitor GPU usage
nvidia-smi

# Reduce model precision
sam_model.half()  # Use 16-bit precision

# Process smaller images
max_image_size = 1024
```

#### 3. Timeout Issues

```python
# Increase timeouts
app.config['PERMANENT_SESSION_LIFETIME'] = 600  # 10 minutes

# For Waitress
serve(app, channel_timeout=600)

# For Gunicorn  
gunicorn --timeout 600 app:app
```

#### 4. CORS Issues

```python
# Configure CORS properly
from flask_cors import CORS

CORS(app, origins=['http://localhost:3000', 'https://yourdomain.com'])
```

### Debugging Tools

#### Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sam_droplet.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use in endpoints
@app.route('/segment_file', methods=['POST'])
def segment_uploaded_file():
    logger.info(f"Processing file: {file.filename}")
    # ... rest of function
```

#### Performance Monitoring

```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@monitor_performance
def segment_image():
    # ... segmentation code
```

### Testing

#### Unit Tests

```python
import unittest
from app import app, process_image_from_base64

class TestSAMAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('status', data)
    
    def test_image_processing(self):
        # Test base64 image processing
        test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        image_array = process_image_from_base64(test_image_b64)
        self.assertEqual(image_array.shape, (1, 1, 3))

if __name__ == '__main__':
    unittest.main()
```

#### Integration Tests

```python
def test_full_segmentation_pipeline():
    """Test complete segmentation workflow."""
    # Load test image
    test_image_path = "test_data/sample_droplets.jpg"
    
    # Analyze image
    with open(test_image_path, 'rb') as f:
        response = app.test_client().post('/analyze_image', 
                                        data={'file': f})
    assert response.status_code == 200
    
    # Segment with filters
    filters = {"mean_min": 140, "area_min": 50}
    with open(test_image_path, 'rb') as f:
        response = app.test_client().post('/segment_file',
                                        data={'file': f, 
                                             'filters': json.dumps(filters)})
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] == True
    assert len(data['masks']) > 0
```

This implementation guide provides a comprehensive overview of the SAM Droplet Segmentation system's architecture and how to extend it. For specific implementation questions, refer to the individual module documentation and code comments. 