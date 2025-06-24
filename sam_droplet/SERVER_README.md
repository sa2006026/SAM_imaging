# SAM Droplet Segmentation Server

This is a Flask web application that provides an API for image segmentation using the Segment Anything Model (SAM).

## Features

- **REST API** for image segmentation
- **File upload** support (multipart/form-data)
- **Base64 image** support (JSON)
- **CORS enabled** for frontend integration
- **Windows compatible** with Waitress server
- **Cross-platform** with Gunicorn support for Linux/Mac
- **Simple web interface** for testing

## API Endpoints

### Health Check
```
GET /health
```
Returns server status and model loading state.

### Segment Image (File Upload)
```
POST /segment_file
Content-Type: multipart/form-data

Form data:
- file: Image file (PNG, JPG, etc.)
```

### Segment Image (Base64)
```
POST /segment
Content-Type: application/json

{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### Response Format
```json
{
  "success": true,
  "num_masks": 42,
  "masks": [
    {
      "id": 0,
      "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
      "area": 1234,
      "bbox": [x, y, width, height],
      "stability_score": 0.95
    }
  ]
}
```

## Setup Instructions

### 1. Install Dependencies

```powershell
# Install server dependencies
pip install -r requirements_server.txt

# Or use the startup script
python start_server.py --install
```

### 2. Download SAM Model

Make sure you have the SAM model file at:
```
sam_droplet/model/sam_vit_h_4b8939.pth
```

### 3. Run the Server

#### Development Mode (Flask)
```powershell
# Using Flask development server
python app.py

# Or using startup script
python start_server.py --mode dev
```

#### Production Mode (Windows - Waitress)
```powershell
# Run with Waitress (Windows compatible)
python start_server.py --mode waitress

# Or run directly
python waitress_config.py
```

#### Production Mode (Linux/Mac - Gunicorn)
```bash
# Run with Gunicorn (Linux/Mac only)
python start_server.py --mode gunicorn
```

### 4. Test the Server

Open your browser and go to:
```
http://localhost:5000
```

Or test the API directly:
```powershell
# Health check
curl http://localhost:5000/health

# Upload image
curl -X POST -F "file=@path/to/image.png" http://localhost:5000/segment_file
```

## Configuration

### Waitress Settings (Windows)

Key settings in `waitress_config.py`:

- **threads**: Number of threads (default: 4)
- **connection_limit**: Max concurrent connections (default: 100)
- **channel_timeout**: Request timeout in seconds (default: 300)
- **max_request_body_size**: Maximum upload size in bytes (default: 100MB)

### Performance Considerations

1. **GPU Usage**: The server will automatically use CUDA if available
2. **Memory**: SAM model requires significant GPU/CPU memory
3. **Timeouts**: Large images may take longer to process
4. **Concurrent Requests**: Limited by available GPU memory

## Frontend Integration

### JavaScript Example

```javascript
// Upload file
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:5000/segment_file', {
    method: 'POST',
    body: formData
});

const result = await response.json();
if (result.success) {
    // Display masks
    result.masks.forEach(mask => {
        const img = document.createElement('img');
        img.src = mask.image;
        document.body.appendChild(img);
    });
}
```

### Base64 Example

```javascript
// Send base64 image
const response = await fetch('http://localhost:5000/segment', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        image: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...'
    })
});
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure SAM model is in `model/sam_vit_h_4b8939.pth`
2. **CUDA errors**: Check GPU memory and CUDA installation
3. **Timeout errors**: Increase `channel_timeout` in waitress_config.py
4. **Upload size**: Increase `max_request_body_size` in waitress_config.py
5. **CORS errors**: Server includes CORS headers, check frontend URL

### Logs

- **Development**: Logs appear in console
- **Waitress**: Logs appear in console (can be redirected to file)

### Performance Tips

1. Use GPU if available (much faster)
2. Resize large images before upload
3. Adjust Waitress thread settings based on hardware
4. Consider image preprocessing for better results

## Platform-Specific Notes

### Windows
- Use **Waitress** for production deployment
- uWSGI is not supported on Windows
- All features work normally

### Linux/Mac
- Can use either **Waitress** or **Gunicorn**
- Gunicorn generally has better performance on Unix systems
- uWSGI can be used but requires additional setup

## Deployment

For production deployment:

### Windows
1. Use Waitress with proper configuration
2. Set up reverse proxy (IIS/nginx)
3. Configure SSL/HTTPS
4. Set up proper logging and monitoring

### Linux/Mac
1. Use Gunicorn or Waitress
2. Set up reverse proxy (nginx/Apache)
3. Configure SSL/HTTPS
4. Set up proper logging and monitoring
5. Consider systemd service for auto-restart 