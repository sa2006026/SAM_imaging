# SAM Droplet Advanced Analysis Server

## Overview

This enhanced version of the SAM Droplet server includes comprehensive mask analysis capabilities, integrating the functionality from your `mask_size_grouping.py` script into a web service.

## New Features

### Advanced Mask Analysis (`/advanced_analysis`)
- **Comprehensive Filtering**: Edge proximity, circularity, and blob distance filtering
- **K-means Clustering**: Intelligent grouping of masks based on 6 features
- **Rich Visualizations**: Cluster overviews and filtered mask visualizations
- **Detailed Statistics**: Processing summaries and cluster statistics
- **CSV Export**: Complete data for all masks with cluster assignments

### Download Analysis (`/download_analysis`)
- **ZIP Package**: Complete analysis results in a downloadable package
- **Multiple Formats**: CSV data, JSON metadata, PNG visualizations
- **Comprehensive Documentation**: README file explaining all outputs

## API Endpoints

### 1. Advanced Analysis
```http
POST /advanced_analysis
Content-Type: application/json
X-API-Key: your-api-key

{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
    "advanced_filters": {
        "edge_threshold": 5,
        "min_circularity": 0.53,
        "max_blob_distance": 50
    },
    "preprocessing": {
        "gaussian_blur": false,
        "contrast_enhancement": false
    }
}
```

**Response:**
```json
{
    "success": true,
    "analysis_results": {
        "processing_summary": {
            "original_masks": 234,
            "after_edge_filtering": 198,
            "after_circularity_filtering": 145,
            "after_blob_filtering": 128,
            "removed_edge_masks": 36,
            "removed_low_circularity_masks": 53,
            "removed_distant_blob_masks": 17
        },
        "filtering_criteria": {
            "edge_threshold": 5,
            "min_circularity": 0.53,
            "max_blob_distance": 50
        },
        "clustering_info": {
            "method": "K-means",
            "n_clusters": 2,
            "cluster_sizes": [89, 39]
        },
        "cluster_statistics": [...]
    },
    "visualizations": {
        "cluster_overview": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        "filtered_edge": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        "filtered_circularity": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    },
    "clustered_masks": {
        "cluster_0": [...],
        "cluster_1": [...]
    },
    "filtered_out_masks": {
        "filtered_edge": [...],
        "filtered_circularity": [...],
        "filtered_distant_blob": [...]
    },
    "csv_data": [...]
}
```

### 2. Download Analysis Package
```http
POST /download_analysis
Content-Type: application/json
X-API-Key: your-api-key

{
    "csv_data": [...],
    "analysis_results": {...},
    "visualizations": {...}
}
```

**Response:** ZIP file download containing:
- `masks_analysis_summary.csv` - Complete mask data
- `analysis_metadata.json` - Detailed analysis results
- `cluster_overview.png` - Visualization of clustered masks
- `filtered_*.png` - Visualizations of filtered masks
- `README.txt` - Documentation

## Web Interface

Access the advanced analysis interface at:
```
http://localhost:9487/static/advanced_analysis.html
```

### Features:
- **Drag & Drop Upload**: Easy image uploading
- **Parameter Control**: Adjust filtering parameters
- **Real-time Results**: Interactive visualization viewing
- **Cluster Navigation**: Browse masks by cluster
- **One-click Download**: Complete analysis package

## Installation & Setup

### 1. Install Dependencies
```bash
cd sam_droplet
pip install -r requirements_server.txt
```

### 2. Download SAM Models
Place your SAM model files in the `model/` directory:
- `sam_vit_h_model.pth` (for vit_h)
- `sam_vit_l_model.pth` (for vit_l)
- `sam_vit_b_model.pth` (for vit_b)

### 3. Start the Server
```bash
python app.py
```

Or using the startup script:
```bash
python start_server.py
```

## Configuration

### Environment Variables
```bash
# API Keys (format: key:name:rate_limit)
API_KEYS=sam-demo-key-123:Demo:50,sam-admin-key-456:Admin:200

# Admin access
ADMIN_API_KEY=admin-secret-key-change-me

# Health check
HEALTH_CHECK_KEY=health-check-key-123
```

### Advanced Filtering Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `edge_threshold` | Distance from edge to consider "touching" | 5 | 0-50 pixels |
| `min_circularity` | Minimum shape circularity (1.0 = perfect circle) | 0.53 | 0.0-1.0 |
| `max_blob_distance` | Maximum distance between blobs in same mask | 50 | 0-200 pixels |

## Clustering Features

The advanced analysis uses K-means clustering with 6 features:
1. **Area**: Mask area in pixels
2. **Bbox Width**: Bounding box width
3. **Bbox Height**: Bounding box height
4. **Aspect Ratio**: Width/height ratio
5. **Stability Score**: SAM confidence score
6. **Circularity**: Shape circularity metric

## Output Data Structure

### CSV Columns
- `mask_id`: Original SAM mask ID
- `area`: Mask area in pixels
- `circularity`: Shape circularity (0-1)
- `blob_distance`: Distance between multiple blobs
- `stability_score`: SAM stability score
- `bbox_*`: Bounding box coordinates and dimensions
- `aspect_ratio`: Width/height ratio
- `cluster_id`: Assigned cluster (0, 1, or -1 if filtered)
- `edge_touching`: Boolean flag for edge filtering
- `low_circularity`: Boolean flag for circularity filtering
- `distant_blob`: Boolean flag for blob distance filtering
- `included_in_clustering`: Boolean flag for final inclusion

## Performance Considerations

- **GPU Acceleration**: Automatically uses CUDA if available
- **Memory Management**: Efficient processing for large images
- **Batch Processing**: Optimized for multiple mask analysis
- **Caching**: Results cached during session for fast re-download

## Comparison with Original Script

| Feature | Original `mask_size_grouping.py` | Server Implementation |
|---------|----------------------------------|----------------------|
| **Input** | Local image file | Base64 image data via API |
| **Output** | Local files (PNG, JSON, CSV) | JSON response + downloadable ZIP |
| **Interface** | Command line | Web interface + REST API |
| **Deployment** | Single-use script | Multi-user web service |
| **Scalability** | One image at a time | Concurrent requests |
| **Integration** | Standalone | Part of larger SAM ecosystem |

## API Authentication

All endpoints require API key authentication:
```http
X-API-Key: your-api-key
```

Default demo key: `sam-demo-key-123`

## Error Handling

The server provides detailed error messages for:
- Invalid image formats
- Missing required parameters
- Processing failures
- Authentication errors
- Rate limiting

## Support

For issues or questions about the advanced analysis functionality:
1. Check the server logs for detailed error messages
2. Verify your API key and parameters
3. Ensure SAM models are properly installed
4. Review the filtering parameters for your use case 