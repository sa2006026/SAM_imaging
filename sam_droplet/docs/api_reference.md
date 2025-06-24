# SAM Droplet Segmentation API Reference

## Overview

The SAM Droplet Segmentation API provides endpoints for image segmentation using the Segment Anything Model (SAM). It supports both file upload and base64 image processing with advanced filtering capabilities.

**Base URL**: `http://localhost:9487` (default)  
**API Version**: 1.0  
**Content Types**: `application/json`, `multipart/form-data`

## Authentication

Currently, no authentication is required. CORS is enabled for cross-origin requests.

## Common Response Format

All API responses follow this general structure:

```json
{
  "success": boolean,
  "error": "string (only present on failure)",
  "data": {}
}
```

## Endpoints

### 1. Health Check

Check server status and model loading state.

**Endpoint**: `GET /health`

#### Response

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Status Codes
- `200`: Server is healthy
- `500`: Server error

---

### 2. Image Segmentation (File Upload)

Segment an uploaded image file and return segmentation masks.

**Endpoint**: `POST /segment_file`  
**Content-Type**: `multipart/form-data`

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Image file (PNG, JPG, JPEG, WebP, etc.) |
| `filters` | String | No | JSON string of filter criteria |

#### Filter Parameters (JSON)

All filter parameters are optional:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `mean_min` | Float | 0-255 | Minimum mean pixel intensity |
| `mean_max` | Float | 0-255 | Maximum mean pixel intensity |
| `min_threshold` | Integer | 0-255 | Exclude objects with pixels below this |
| `max_threshold` | Integer | 0-255 | Exclude objects with pixels above this |
| `std_min` | Float | 0+ | Minimum standard deviation (texture) |
| `std_max` | Float | 0+ | Maximum standard deviation |
| `area_min` | Integer | 0+ | Minimum object area in pixels |
| `area_max` | Integer | 0+ | Maximum object area in pixels |
| `median_min` | Float | 0-255 | Minimum median pixel intensity |
| `median_max` | Float | 0-255 | Maximum median pixel intensity |
| `min_edge_distance` | Integer | 0+ | Minimum distance from image edge |
| `exclude_edge_touching` | Boolean | - | Exclude objects touching edges |

#### Example Request

```bash
curl -X POST http://localhost:9487/segment_file \
  -F "file=@droplets.jpg" \
  -F 'filters={"mean_min": 140, "area_min": 50, "area_max": 5000}'
```

#### Response

```json
{
  "success": true,
  "filename": "droplets.jpg",
  "num_masks": 42,
  "masks": [
    {
      "id": 0,
      "image": "data:image/png;base64,iVBORw0KGgoAAAANSU...",
      "area": 1234,
      "bbox": [x, y, width, height],
      "stability_score": 0.95,
      "pixel_stats": {
        "mean_intensity": 165.3,
        "min_intensity": 120,
        "max_intensity": 200,
        "std_intensity": 15.2,
        "median_intensity": 168.0,
        "pixel_count": 1234
      },
      "edge_stats": {
        "min_distance_to_edge": 45,
        "touches_edge": false,
        "distances": {
          "top": 120,
          "bottom": 45,
          "left": 89,
          "right": 156
        },
        "edge_touching_sides": []
      }
    }
  ],
  "filters_applied": true
}
```

#### Status Codes
- `200`: Success
- `400`: No file provided or invalid filters
- `500`: Processing error

---

### 3. Image Segmentation (Base64)

Segment a base64-encoded image.

**Endpoint**: `POST /segment`  
**Content-Type**: `application/json`

#### Request Body

```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSU...",
  "filters": {
    "mean_min": 140,
    "area_min": 50
  }
}
```

#### Response

Same format as `/segment_file` endpoint.

#### Example Request

```bash
curl -X POST http://localhost:9487/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSU...",
    "filters": {"mean_min": 140, "area_min": 50}
  }'
```

---

### 4. Image Analysis

Analyze an image to get pixel statistics for filter setup.

**Endpoint**: `POST /analyze_image`  
**Content-Type**: `multipart/form-data`

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Image file to analyze |

#### Response

```json
{
  "success": true,
  "filename": "droplets.jpg",
  "analysis": {
    "overall_stats": {
      "mean_intensity": 128.5,
      "min_intensity": 12,
      "max_intensity": 245,
      "std_intensity": 45.2,
      "median_intensity": 125.0,
      "histogram": [120, 340, 560, ...]
    },
    "image_info": {
      "width": 1920,
      "height": 1080,
      "channels": 3,
      "total_pixels": 2073600
    }
  }
}
```

#### Example Request

```bash
curl -X POST http://localhost:9487/analyze_image \
  -F "file=@droplets.jpg"
```

---

### 5. Static File Serving

Serve the web interface and static assets.

**Endpoint**: `GET /`  
**Response**: HTML web interface

**Endpoint**: `GET /static/<filename>`  
**Response**: Static files (CSS, JS, images)

## Error Handling

### Common Error Responses

```json
{
  "error": "Error description",
  "success": false
}
```

### Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters or missing data |
| 500 | Internal Server Error - Processing or model error |

### Common Errors

1. **No file provided**
   ```json
   {"error": "No file uploaded", "success": false}
   ```

2. **Model not loaded**
   ```json
   {"error": "SAM model not initialized", "success": false}
   ```

3. **Invalid image format**
   ```json
   {"error": "Could not process image format", "success": false}
   ```

4. **GPU memory error**
   ```json
   {"error": "CUDA out of memory", "success": false}
   ```

## Rate Limiting

Currently no rate limiting is implemented. Consider implementing rate limiting for production use.

## Data Types

### Mask Object

```typescript
interface Mask {
  id: number;                    // Unique mask identifier
  image: string;                 // Base64 encoded PNG mask
  area: number;                  // Area in pixels
  bbox: [number, number, number, number]; // [x, y, width, height]
  stability_score: number;       // SAM confidence score (0-1)
  pixel_stats?: PixelStats;      // Present when filters applied
  edge_stats?: EdgeStats;        // Present when filters applied
}
```

### Pixel Statistics

```typescript
interface PixelStats {
  mean_intensity: number;        // Average pixel value (0-255)
  min_intensity: number;         // Minimum pixel value (0-255)
  max_intensity: number;         // Maximum pixel value (0-255)
  std_intensity: number;         // Standard deviation of pixels
  median_intensity: number;      // Median pixel value (0-255)
  pixel_count: number;           // Number of pixels in mask
}
```

### Edge Statistics

```typescript
interface EdgeStats {
  min_distance_to_edge: number;          // Minimum distance to any edge
  touches_edge: boolean;                 // Whether mask touches any edge
  distances: {                           // Distance to each edge
    top: number;
    bottom: number;
    left: number;
    right: number;
  };
  edge_touching_sides: string[];         // Which edges are touched
}
```

## Best Practices

### Performance

1. **Image Size**: Resize large images before upload for faster processing
2. **GPU Usage**: Ensure CUDA is available for optimal performance
3. **Batch Processing**: Process multiple images sequentially rather than concurrently

### Filtering

1. **Use Analysis**: Call `/analyze_image` first to understand pixel distribution
2. **Start Broad**: Begin with loose filters and tighten as needed
3. **Combine Filters**: Use multiple filter types for better precision

### Error Handling

1. **Check Health**: Verify model is loaded before processing
2. **Validate Input**: Ensure image files are valid before upload
3. **Handle Timeouts**: Large images may take time to process

## SDK Examples

### Python Client

```python
import requests
import base64

class SAMClient:
    def __init__(self, base_url="http://localhost:9487"):
        self.base_url = base_url
    
    def health_check(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def segment_file(self, image_path, filters=None):
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {}
            if filters:
                data['filters'] = json.dumps(filters)
            
            response = requests.post(
                f"{self.base_url}/segment_file",
                files=files,
                data=data
            )
            return response.json()
    
    def analyze_image(self, image_path):
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/analyze_image",
                files=files
            )
            return response.json()

# Usage
client = SAMClient()
health = client.health_check()
analysis = client.analyze_image("droplets.jpg")
result = client.segment_file("droplets.jpg", {
    "mean_min": 140,
    "area_min": 50
})
```

### JavaScript Client

```javascript
class SAMClient {
    constructor(baseURL = 'http://localhost:9487') {
        this.baseURL = baseURL;
    }

    async healthCheck() {
        const response = await fetch(`${this.baseURL}/health`);
        return await response.json();
    }

    async segmentFile(file, filters = null) {
        const formData = new FormData();
        formData.append('file', file);
        if (filters) {
            formData.append('filters', JSON.stringify(filters));
        }

        const response = await fetch(`${this.baseURL}/segment_file`, {
            method: 'POST',
            body: formData
        });
        return await response.json();
    }

    async segmentBase64(imageData, filters = null) {
        const response = await fetch(`${this.baseURL}/segment`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                filters: filters
            })
        });
        return await response.json();
    }

    async analyzeImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.baseURL}/analyze_image`, {
            method: 'POST',
            body: formData
        });
        return await response.json();
    }
}

// Usage
const client = new SAMClient();
const health = await client.healthCheck();
const analysis = await client.analyzeImage(fileInput.files[0]);
const result = await client.segmentFile(fileInput.files[0], {
    mean_min: 140,
    area_min: 50
});