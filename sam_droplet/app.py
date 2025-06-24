"""Flask web application for SAM droplet segmentation."""

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
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Import authentication middleware
from auth_middleware import (
    require_api_key, 
    validate_file_upload, 
    track_usage, 
    require_admin_key,
    get_usage_stats
)

# Import our filtering utilities
import sys
sys.path.append(str(Path(__file__).parent / "src"))
from sam_droplet.filters import (
    analyze_mask_pixels, 
    analyze_image_statistics, 
    filter_masks_by_criteria,
    analyze_edge_proximity,
    preprocess_image,
    get_default_filters,
    get_default_preprocessing,
    apply_preprocessing_filters,
    calculate_summary_statistics
)

# Import advanced mask analysis
from mask_analysis_server import mask_analyzer

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

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
        
        # Try to find available SAM models (prioritize vit_h like mask_size_grouping.py)
        model_files = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth", 
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        
        # Check for existing models - prioritize vit_h (same as mask_size_grouping.py)
        available_model = None
        model_type = None
        
        # First check for vit_h (preferred model from your script)
        for mtype in ["vit_h", "vit_l", "vit_b"]:  # Priority order
            filename = model_files[mtype]
            model_path = project_root / "model" / filename
            if model_path.exists():
                available_model = model_path
                model_type = mtype
                print(f"Found SAM model: {mtype}")
                break
        
        # If no standard SAM models found, download vit_b (smallest)
        if not available_model:
            print("No SAM models found. Downloading SAM vit_b model...")
            import urllib.request
            
            model_type = "vit_b"
            model_filename = model_files[model_type]
            model_path = project_root / "model" / model_filename
            model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            
            # Create model directory if it doesn't exist
            model_path.parent.mkdir(exist_ok=True)
            
            print(f"Downloading {model_filename}...")
            urllib.request.urlretrieve(model_url, str(model_path))
            print("Download completed!")
            available_model = model_path
        
        # Initialize model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(f"Loading SAM model: {model_type} from {available_model}")
        
        sam_model = sam_model_registry[model_type](checkpoint=str(available_model))
        sam_model.to(device=device)
        
        # Create mask generator with exact same parameters as mask_size_grouping.py
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=32,
            crop_n_layers=3,  # This is the key parameter from your script
        )
        print("SAM model initialized successfully!")
        print(f"Mask generator configured with: points_per_side=32, crop_n_layers=3")

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
            # Add Feret diameter as a top-level field for easy access
            if 'max_feret_diameter' in mask['pixel_stats']:
                mask_data['max_feret_diameter'] = float(mask['pixel_stats']['max_feret_diameter'])
        
        # Add edge statistics if available
        if 'edge_stats' in mask:
            mask_data['edge_stats'] = convert_numpy_types(mask['edge_stats'])
        
        mask_images.append(mask_data)
    
    return mask_images

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': sam_model is not None})

@app.route('/')
def index():
    """Serve the main HTML interface."""
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

@app.route('/segment', methods=['POST'])
@require_api_key
@track_usage
def segment_image():
    """Segment an uploaded image and return masks."""
    try:
        # Initialize SAM if not already done
        initialize_sam()
        
        # Get image data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process the image
        image_array = process_image_from_base64(data['image'])
        
        print(f"Processing image of shape: {image_array.shape}")
        
        # Apply image preprocessing if requested (blur, contrast enhancement, etc.)
        preprocessing_options = data.get('preprocessing', get_default_preprocessing())
        if preprocessing_options:
            print(f"Applying image preprocessing: {preprocessing_options}")
            image_array = preprocess_image(image_array, preprocessing_options)
        
        # Generate masks using SAM
        everything_results = mask_generator.generate(image_array)
        
        print(f"SAM generated {len(everything_results)} initial masks")
        
        # Add pixel statistics and edge analysis to all masks first
        for mask in everything_results:
            mask['pixel_stats'] = analyze_mask_pixels(image_array, mask['segmentation'])
            mask['edge_stats'] = analyze_edge_proximity(mask['segmentation'], image_array.shape)
        
        # Store original masks for frontend display
        original_masks_for_frontend = masks_to_base64_images(everything_results)
        
        # Apply automatic preprocessing filters (50% above average area + edge-touching removal)
        preprocessed_masks = apply_preprocessing_filters(everything_results, image_array.shape)
        
        # Get filter criteria from request (for additional user-defined filtering)
        filter_criteria = data.get('filters', {})
        
        # Apply additional user-defined filters if any
        if filter_criteria:
            print(f"Applying additional user filters: {filter_criteria}")
            filtered_masks = filter_masks_by_criteria(
                preprocessed_masks, 
                image_array.shape,
                filter_criteria
            )
            print(f"After user filtering: {len(filtered_masks)} masks remaining")
        else:
            # Apply default user filters (minimum area, etc.)
            default_filters = get_default_filters(image_array.shape)
            filtered_masks = filter_masks_by_criteria(
                preprocessed_masks,
                image_array.shape, 
                default_filters
            )
            print(f"After default user filtering: {len(filtered_masks)} masks remaining")
        
        # Convert to base64 images
        mask_images = masks_to_base64_images(filtered_masks)
        
        # Calculate summary statistics
        intensity_threshold = filter_criteria.get('intensity_threshold') if filter_criteria.get('enable_intensity_coloring') else None
        summary_stats = calculate_summary_statistics(filtered_masks, intensity_threshold)
        
        # Calculate statistical info for response
        original_count = len(everything_results)
        preprocessed_count = len(preprocessed_masks)
        final_count = len(filtered_masks)
        preprocessing_removed = original_count - preprocessed_count
        user_filter_removed = preprocessed_count - final_count
        
        # Sort by area (largest first)
        mask_images.sort(key=lambda x: x['area'], reverse=True)
        
        return jsonify({
            'success': True,
            'num_masks': len(mask_images),
            'original_count': original_count,
            'preprocessing_removed': preprocessing_removed,
            'user_filter_removed': user_filter_removed,
            'preprocessing_info': {
                'removed_large_masks': 'Masks >50% larger than average',
                'removed_edge_touching': 'Masks touching image edges',
                'removed_non_circular': 'Masks with circularity < 0.85'
            },
            'masks': mask_images,
            'filters_applied': filter_criteria,
            'preprocessing_applied': preprocessing_options,
            'summary_statistics': summary_stats,
            'original_masks_for_frontend': original_masks_for_frontend
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
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
        
        # Get preprocessing options from form data (if available)
        preprocessing_options = {}
        if request.form.get('preprocessing'):
            try:
                preprocessing_options = json.loads(request.form.get('preprocessing'))
            except json.JSONDecodeError:
                preprocessing_options = get_default_preprocessing()
        else:
            preprocessing_options = get_default_preprocessing()
        
        # Apply image preprocessing if requested (blur, contrast enhancement, etc.)
        if preprocessing_options:
            print(f"Applying image preprocessing: {preprocessing_options}")
            image_array = preprocess_image(image_array, preprocessing_options)
        
        # Generate masks
        everything_results = mask_generator.generate(image_array)
        
        print(f"SAM generated {len(everything_results)} initial masks")
        
        # Add pixel statistics and edge analysis to all masks first
        for mask in everything_results:
            mask['pixel_stats'] = analyze_mask_pixels(image_array, mask['segmentation'])
            mask['edge_stats'] = analyze_edge_proximity(mask['segmentation'], image_array.shape)
        
        # Store original masks for frontend display
        original_masks_for_frontend = masks_to_base64_images(everything_results)
        
        # Apply automatic preprocessing filters (50% above average area + edge-touching removal)
        preprocessed_masks = apply_preprocessing_filters(everything_results, image_array.shape)
        
        # Get filter criteria from form data (for additional user-defined filtering)
        filter_criteria = {}
        if request.form.get('filters'):
            try:
                filter_criteria = json.loads(request.form.get('filters'))
            except json.JSONDecodeError:
                filter_criteria = {}
        
        # Apply additional user-defined filters if any
        if filter_criteria:
            print(f"Applying additional user filters: {filter_criteria}")
            filtered_masks = filter_masks_by_criteria(
                preprocessed_masks, 
                image_array.shape,
                filter_criteria
            )
            print(f"After user filtering: {len(filtered_masks)} masks remaining")
        else:
            # Apply default user filters (minimum area, etc.)
            default_filters = get_default_filters(image_array.shape)
            filtered_masks = filter_masks_by_criteria(
                preprocessed_masks,
                image_array.shape, 
                default_filters
            )
            print(f"After default user filtering: {len(filtered_masks)} masks remaining")
        
        # Convert to base64 images
        mask_images = masks_to_base64_images(filtered_masks)
        
        # Calculate summary statistics
        intensity_threshold = filter_criteria.get('intensity_threshold') if filter_criteria.get('enable_intensity_coloring') else None
        summary_stats = calculate_summary_statistics(filtered_masks, intensity_threshold)
        
        # Calculate statistical info for response
        original_count = len(everything_results)
        preprocessed_count = len(preprocessed_masks)
        final_count = len(filtered_masks)
        preprocessing_removed = original_count - preprocessed_count
        user_filter_removed = preprocessed_count - final_count
        
        # Sort by area (largest first)
        mask_images.sort(key=lambda x: x['area'], reverse=True)
        
        return jsonify({
            'success': True,
            'num_masks': len(mask_images),
            'original_count': original_count,
            'preprocessing_removed': preprocessing_removed,
            'user_filter_removed': user_filter_removed,
            'preprocessing_info': {
                'removed_large_masks': 'Masks >50% larger than average',
                'removed_edge_touching': 'Masks touching image edges',
                'removed_non_circular': 'Masks with circularity < 0.85'
            },
            'masks': mask_images,
            'filters_applied': filter_criteria,
            'preprocessing_applied': preprocessing_options,
            'summary_statistics': summary_stats,
            'original_masks_for_frontend': original_masks_for_frontend
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
        
        # Convert to numpy array and grayscale
        image_array = np.array(pil_image)
        
        # Calculate overall image statistics using the new function
        image_stats = analyze_image_statistics(image_array)
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'analysis': image_stats
        })
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/defaults', methods=['GET'])
def get_defaults():
    """Get default filter and preprocessing settings."""
    try:
        # Use a default image shape for generating defaults
        default_shape = (1024, 1024, 3)
        
        defaults = {
            'filters': get_default_filters(default_shape),
            'preprocessing': get_default_preprocessing()
        }
        
        return jsonify({
            'success': True,
            'defaults': defaults,
            'description': {
                'automatic_preprocessing': {
                    'description': 'Always applied during mask generation',
                    'area_outlier_removal': 'Removes masks >50% larger than average area',
                    'edge_touching_removal': 'Removes masks touching image edges (0 pixel distance)',
                    'circularity_filter': 'Removes masks with circularity < 0.85 (non-circular shapes)',
                    'purpose': 'Eliminates large artifacts, incomplete objects, and non-circular shapes'
                },
                'filters': {
                    'area_max': 'Additional maximum area limit for user filtering (15000 pixels)',
                    'area_min': 'Minimum area to filter tiny artifacts (50 pixels)',
                    'mean_min/max': 'Filter by average pixel intensity',
                    'std_min/max': 'Filter by texture/variation within objects'
                },
                'preprocessing': {
                    'gaussian_blur': 'Apply blur for noise reduction before segmentation',
                    'contrast_enhancement': 'Enhance contrast using CLAHE',
                    'purpose': 'Improve segmentation quality for difficult images'
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get defaults: {str(e)}'
        }), 500

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

@app.route('/advanced_analysis', methods=['POST'])
@require_api_key
@track_usage
def advanced_mask_analysis():
    """
    Perform advanced mask analysis with filtering, clustering, and comprehensive reporting.
    Similar to the mask_size_grouping.py functionality but as a web service.
    """
    try:
        # Initialize SAM if not already done
        initialize_sam()
        
        # Get data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process the image
        image_array = process_image_from_base64(data['image'])
        
        print(f"Processing image for advanced analysis: {image_array.shape}")
        
        # Apply image preprocessing if requested
        preprocessing_options = data.get('preprocessing', {})
        if preprocessing_options:
            print(f"Applying image preprocessing: {preprocessing_options}")
            image_array = preprocess_image(image_array, preprocessing_options)
        
        # Generate masks using SAM
        everything_results = mask_generator.generate(image_array)
        
        print(f"SAM generated {len(everything_results)} initial masks for advanced analysis")
        print(f"Using mask_size_grouping.py configuration: vit_h model, crop_n_layers=3, points_per_side=32")
        
        # Add pixel statistics, edge analysis, and circularity to all masks
        for i, mask in enumerate(everything_results):
            mask['pixel_stats'] = analyze_mask_pixels(image_array, mask['segmentation'])
            mask['edge_stats'] = analyze_edge_proximity(mask['segmentation'], image_array.shape)
            
            # Add circularity using the existing function
            if 'circularity' not in mask:
                from sam_droplet.filters import calculate_circularity
                mask['circularity'] = calculate_circularity(mask['segmentation'])
        
        # Get filtering parameters from request (match mask_size_grouping.py defaults)
        filtering_params = data.get('advanced_filters', {
            'edge_threshold': 5,
            'min_circularity': 0.53,
            'max_blob_distance': 80,  # Updated to match your script's latest config
            'overlap_min_percentage': 80.0  # NEW: Minimum overlap percentage for analysis
        })
        
        print(f"Applying advanced filtering with params: {filtering_params}")
        
        # Perform comprehensive analysis using our advanced analyzer
        analysis_results = mask_analyzer.create_comprehensive_analysis(
            everything_results, 
            image_array.shape,
            filtering_params
        )
        
        # Create visualizations
        visualizations = mask_analyzer.create_visualization_summary(image_array)
        
        # Create CSV data for download
        csv_data = mask_analyzer.create_summary_csv_data()
        
        # Convert individual masks to base64 for frontend display
        clustered_mask_images = {}
        for cluster_id, cluster in enumerate(mask_analyzer.clustered_masks):
            if cluster:
                clustered_mask_images[f'cluster_{cluster_id}'] = masks_to_base64_images(cluster)
        
        # Also include filtered out masks
        filtered_out_images = {}
        for category, masks in mask_analyzer.removed_masks.items():
            if masks:
                filtered_out_images[f'filtered_{category}'] = masks_to_base64_images(masks)
        
        # Convert numpy types for JSON serialization
        analysis_results = convert_numpy_types(analysis_results)
        csv_data = convert_numpy_types(csv_data)
        
        return jsonify({
            'success': True,
            'analysis_results': analysis_results,
            'visualizations': visualizations,
            'clustered_masks': clustered_mask_images,
            'filtered_out_masks': filtered_out_images,
            'csv_data': csv_data,
            'summary': {
                'total_processing_time': 'Advanced mask analysis completed',
                'clustering_method': 'K-means with 6 features',
                'features_used': ['area', 'bbox_width', 'bbox_height', 'aspect_ratio', 'stability_score', 'circularity'],
                'filtering_applied': ['edge_proximity', 'circularity', 'blob_distance'],
                'overlap_analysis': 'Large masks labeled with count of overlapping small masks (≥80% coverage)',
                'output_format': {
                    'analysis_results': 'Comprehensive statistics and metadata with overlap analysis',
                    'visualizations': 'Base64 encoded visualization images including overlap analysis',
                    'clustered_masks': 'Individual mask data grouped by cluster',
                    'filtered_out_masks': 'Masks that were filtered out during processing',
                    'csv_data': 'CSV-ready data for all masks with filter/cluster/overlap information'
                }
            }
        })
        
    except Exception as e:
        print(f"Error in advanced mask analysis: {str(e)}")
        return jsonify({'error': f'Advanced analysis failed: {str(e)}'}), 500

@app.route('/download_analysis', methods=['POST'])
@require_api_key
@track_usage
def download_analysis_zip():
    """
    Create and return a ZIP file containing comprehensive analysis results.
    Includes CSV data, visualizations, and metadata JSON.
    """
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'csv_data' not in data or 'analysis_results' not in data:
            return jsonify({'error': 'Missing analysis data for download'}), 400
        
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save CSV data
            csv_path = temp_path / 'masks_analysis_summary.csv'
            csv_data = data['csv_data']
            
            if csv_data:
                import csv
                with open(csv_path, 'w', newline='') as csvfile:
                    if isinstance(csv_data[0], dict):
                        fieldnames = csv_data[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(csv_data)
            
            # Save analysis results as JSON
            json_path = temp_path / 'analysis_metadata.json'
            with open(json_path, 'w') as f:
                json.dump(data['analysis_results'], f, indent=2)
            
            # Save visualizations if provided
            visualizations = data.get('visualizations', {})
            for viz_name, viz_base64 in visualizations.items():
                if viz_base64.startswith('data:image/png;base64,'):
                    # Remove the data URL prefix
                    image_data = viz_base64.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    
                    viz_path = temp_path / f'{viz_name}.png'
                    with open(viz_path, 'wb') as f:
                        f.write(image_bytes)
            
            # Create README file
            readme_path = temp_path / 'README.txt'
            with open(readme_path, 'w') as f:
                f.write("""SAM Droplet Advanced Analysis Results
=====================================

This ZIP contains the results of advanced mask analysis:

Files included:
- masks_analysis_summary.csv: Detailed data for all masks with cluster assignments
- analysis_metadata.json: Comprehensive analysis results and statistics
- cluster_overview.png: Visualization of clustered masks
- filtered_*.png: Visualizations of filtered out masks by category
- README.txt: This file

CSV Columns:
- mask_id: Original SAM mask ID
- area: Mask area in pixels
- circularity: Shape circularity (0-1, 1=perfect circle)
- blob_distance: Distance between multiple blobs in mask
- stability_score: SAM stability score
- bbox_*: Bounding box coordinates and dimensions
- cluster_id: Assigned cluster (0, 1, or -1 if filtered out)
- *_touching/*_circularity/*_blob: Boolean flags for filtering reasons

Generated by SAM Droplet Advanced Analysis Server
""")
            
            # Create ZIP file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in temp_path.iterdir():
                    if file_path.is_file():
                        zip_file.write(file_path, file_path.name)
            
            zip_buffer.seek(0)
            
            return send_file(
                zip_buffer,
                as_attachment=True,
                download_name='sam_droplet_advanced_analysis.zip',
                mimetype='application/zip'
            )
    
    except Exception as e:
        print(f"Error creating analysis download: {str(e)}")
        return jsonify({'error': f'Failed to create download: {str(e)}'}), 500

if __name__ == '__main__':
    # Set up environment variables for development
    os.environ.setdefault('API_KEYS', 'sam-demo-key-123:Demo:50,sam-admin-key-456:Admin:200')
    os.environ.setdefault('ADMIN_API_KEY', 'admin-secret-key-change-me')
    os.environ.setdefault('HEALTH_CHECK_KEY', 'health-check-key-123')
    
    # Initialize SAM model on startup
    initialize_sam()
    
    # Run Flask development server
    app.run(host='0.0.0.0', port=9487, debug=True) 