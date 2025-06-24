"""
Example of app.py with authentication added.
This shows how to modify your existing app.py with minimal changes.
"""

import os
import io
import base64
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import torch
from PIL import Image
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Import authentication middleware
from auth_middleware import (
    require_api_key, 
    validate_file_upload, 
    track_usage, 
    require_admin_key,
    get_usage_stats
)

# Import filtering utilities
import sys
sys.path.append(str(Path(__file__).parent / "src"))
from sam_droplet.filters import (
    analyze_mask_pixels, 
    analyze_image_statistics, 
    filter_masks_by_criteria,
    analyze_edge_proximity
)

app = Flask(__name__)

# Configure CORS with more restrictive settings for production
CORS(app, origins=os.getenv('ALLOWED_ORIGINS', '*').split(','))

# Global variables for SAM model
sam_model = None
mask_generator = None

def initialize_sam():
    """Initialize the SAM model."""
    global sam_model, mask_generator
    
    if sam_model is None:
        print("Initializing SAM model...")
        
        # Get paths
        project_root = Path(__file__).parent
        model_path = project_root / "model" / "mobile_sam.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"SAM model not found at: {model_path}")
        
        # Initialize model
        model_type = "vit_t"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        sam_model = sam_model_registry[model_type](checkpoint=str(model_path))
        sam_model.to(device=device)
        
        mask_generator = SamAutomaticMaskGenerator(sam_model)
        print("SAM model initialized successfully!")

def process_image_from_base64(image_data):
    """Convert base64 image to numpy array."""
    # Remove data URL prefix if present
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    
    # Convert to PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    image_array = np.array(pil_image)
    
    return image_array

def masks_to_base64_images(masks):
    """Convert masks to base64 encoded PNG images."""
    mask_images = []
    
    for i, mask in enumerate(masks):
        # Get the segmentation mask
        segmentation = mask['segmentation']
        
        # Convert boolean mask to uint8 (0 or 255)
        mask_image = (segmentation * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_mask = Image.fromarray(mask_image, mode='L')
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_mask.save(buffer, format='PNG')
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        mask_data = {
            'id': i,
            'image': f"data:image/png;base64,{mask_base64}",
            'area': int(mask['area']),
            'bbox': convert_numpy_types(mask['bbox']),
            'stability_score': float(mask['stability_score'])
        }
        
        # Add pixel statistics if available
        if 'pixel_stats' in mask:
            mask_data['pixel_stats'] = convert_numpy_types(mask['pixel_stats'])
        
        # Add edge statistics if available
        if 'edge_stats' in mask:
            mask_data['edge_stats'] = convert_numpy_types(mask['edge_stats'])
        
        mask_images.append(mask_data)
    
    return mask_images

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# ===== ROUTES =====

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - kept open for monitoring."""
    # Optional: Add basic health check key for monitoring services
    health_key = request.headers.get('X-Health-Key')
    expected_key = os.getenv('HEALTH_CHECK_KEY')
    
    basic_health = {
        'status': 'healthy', 
        'model_loaded': sam_model is not None
    }
    
    # Return basic info for unauthenticated requests
    if not expected_key or health_key != expected_key:
        return jsonify(basic_health)
    
    # Return detailed info for authenticated health checks
    detailed_health = basic_health.copy()
    detailed_health.update({
        'gpu_available': torch.cuda.is_available(),
        'timestamp': str(torch.tensor(1).float().item())  # Quick GPU test
    })
    
    if torch.cuda.is_available():
        detailed_health['gpu_memory'] = {
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'cached_mb': torch.cuda.memory_reserved() / 1024 / 1024
        }
    
    return jsonify(detailed_health)

@app.route('/')
def index():
    """Serve the main HTML interface."""
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

# ===== PROTECTED ENDPOINTS =====

@app.route('/segment', methods=['POST'])
@require_api_key
@track_usage
def segment_image():
    """Segment a base64 image and return masks."""
    try:
        # Initialize SAM if not already done
        initialize_sam()
        
        # Get image data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process the image
        image_array = process_image_from_base64(data['image'])
        
        print(f"Processing base64 image of shape: {image_array.shape}")
        
        # Generate masks
        masks = mask_generator.generate(image_array)
        
        print(f"Generated {len(masks)} masks")
        
        # Apply filters if provided
        filters = data.get('filters', {})
        if filters:
            print(f"Applying filters: {filters}")
            masks = filter_masks_by_criteria(masks, image_array, filters)
            print(f"After filtering: {len(masks)} masks")
        else:
            # Add pixel statistics to all masks even without filtering
            for mask in masks:
                mask['pixel_stats'] = analyze_mask_pixels(image_array, mask['segmentation'])
                mask['edge_stats'] = analyze_edge_proximity(mask['segmentation'], image_array.shape)
        
        # Convert masks to base64 images
        mask_images = masks_to_base64_images(masks)
        
        # Sort by area (largest first)
        mask_images.sort(key=lambda x: x['area'], reverse=True)
        
        return jsonify({
            'success': True,
            'num_masks': len(mask_images),
            'masks': mask_images,
            'filters_applied': bool(filters)
        })
        
    except Exception as e:
        print(f"Error processing base64 image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/segment_file', methods=['POST'])
@require_api_key
@validate_file_upload
@track_usage
def segment_uploaded_file():
    """Segment an uploaded file and return masks."""
    try:
        # Initialize SAM if not already done
        initialize_sam()
        
        file = request.files['file']
        
        # Read and process the image
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        print(f"Processing uploaded file: {file.filename}, shape: {image_array.shape}")
        
        # Generate masks
        masks = mask_generator.generate(image_array)
        
        print(f"Generated {len(masks)} masks")
        
        # Get filters from form data (if provided as JSON string)
        filters = {}
        if 'filters' in request.form:
            try:
                filters = json.loads(request.form['filters'])
            except json.JSONDecodeError:
                print("Warning: Could not parse filters from form data")
        
        # Apply filters if provided
        if filters:
            print(f"Applying filters: {filters}")
            masks = filter_masks_by_criteria(masks, image_array, filters)
            print(f"After filtering: {len(masks)} masks")
        else:
            # Add pixel statistics to all masks even without filtering
            for mask in masks:
                mask['pixel_stats'] = analyze_mask_pixels(image_array, mask['segmentation'])
                mask['edge_stats'] = analyze_edge_proximity(mask['segmentation'], image_array.shape)
        
        # Convert masks to base64 images
        mask_images = masks_to_base64_images(masks)
        
        # Sort by area (largest first)
        mask_images.sort(key=lambda x: x['area'], reverse=True)
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'num_masks': len(mask_images),
            'masks': mask_images,
            'filters_applied': bool(filters)
        })
        
    except Exception as e:
        print(f"Error processing uploaded file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_image', methods=['POST'])
@require_api_key
@validate_file_upload
@track_usage
def analyze_image():
    """Analyze an image and return pixel statistics to help with filter setup."""
    try:
        file = request.files['file']
        
        # Read and process the image
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Calculate overall image statistics
        image_stats = analyze_image_statistics(image_array)
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'analysis': image_stats
        })
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ===== ADMIN ENDPOINTS =====

@app.route('/admin/stats', methods=['GET'])
@require_admin_key
def admin_stats():
    """Get usage statistics (admin only)."""
    return jsonify(get_usage_stats())

@app.route('/admin/health', methods=['GET'])
@require_admin_key
def admin_health():
    """Detailed health check (admin only)."""
    health_data = {
        'status': 'healthy',
        'model_loaded': sam_model is not None,
        'gpu_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        health_data['gpu_info'] = {
            'device_name': torch.cuda.get_device_name(0),
            'memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'memory_total_mb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        }
    
    return jsonify(health_data)

if __name__ == '__main__':
    # Set up environment variables
    os.environ.setdefault('API_KEYS', 'sam-demo-key-123:Demo:50,sam-admin-key-456:Admin:200')
    os.environ.setdefault('ADMIN_API_KEY', 'admin-secret-key-change-me')
    os.environ.setdefault('HEALTH_CHECK_KEY', 'health-check-key-123')
    
    # Initialize SAM model on startup
    initialize_sam()
    
    # Run Flask development server
    app.run(host='0.0.0.0', port=9487, debug=False) 