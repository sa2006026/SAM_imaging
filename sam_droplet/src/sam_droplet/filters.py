"""
Pixel-level filtering utilities for SAM droplet segmentation.

This module provides functions for analyzing and filtering segmentation masks
based on pixel intensity characteristics and spatial proximity to image edges.
"""

import numpy as np
import cv2
from typing import Dict, List, Any, Optional
from scipy.spatial.distance import pdist, squareform
from skimage import morphology, filters


def calculate_max_feret_diameter(mask: np.ndarray) -> float:
    """
    Calculate the maximum Feret diameter of a mask.
    
    The Feret diameter is the distance between the two parallel planes 
    restricting the object perpendicular to that direction.
    The maximum Feret diameter is the longest distance between any two points 
    on the boundary of the object.
    
    Args:
        mask: Boolean mask array
        
    Returns:
        Maximum Feret diameter in pixels
    """
    # Find contours of the mask
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return 0.0
    
    # Get the largest contour (should be the main object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) < 2:
        return 0.0
    
    # Convert contour points to 2D array
    points = largest_contour.reshape(-1, 2)
    
    # For performance, if there are too many points, sample them
    if len(points) > 200:
        # Sample points evenly along the contour
        indices = np.linspace(0, len(points) - 1, 200, dtype=int)
        points = points[indices]
    
    # Calculate all pairwise distances
    if len(points) > 1:
        distances = pdist(points, metric='euclidean')
        max_feret = np.max(distances)
    else:
        max_feret = 0.0
    
    return float(max_feret)


def calculate_circularity(mask: np.ndarray) -> float:
    """
    Calculate the circularity of a mask.
    
    Circularity is defined as: 4π * Area / Perimeter²
    A perfect circle has circularity = 1.0
    Values closer to 1.0 indicate more circular shapes
    
    Args:
        mask: Boolean mask array
        
    Returns:
        Circularity value (0.0 to 1.0)
    """
    # Find contours of the mask
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return 0.0
    
    # Get the largest contour (should be the main object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate area and perimeter
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    if perimeter == 0 or area == 0:
        return 0.0
    
    # Calculate circularity: 4π * Area / Perimeter²
    circularity = (4 * np.pi * area) / (perimeter * perimeter)
    
    # Ensure the value is between 0 and 1
    return min(1.0, max(0.0, float(circularity)))


def preprocess_image(image: np.ndarray, preprocessing_options: Dict[str, Any]) -> np.ndarray:
    """
    Apply preprocessing filters to an image before segmentation.
    
    Args:
        image: Input image (RGB or grayscale)
        preprocessing_options: Dictionary of preprocessing options
        
    Returns:
        Preprocessed image
    """
    processed_image = image.copy()
    
    # Convert to grayscale for some operations if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image.copy()
    
    # Gaussian blur for noise reduction
    if preprocessing_options.get('gaussian_blur', False):
        kernel_size = preprocessing_options.get('gaussian_kernel_size', 3)
        sigma = preprocessing_options.get('gaussian_sigma', 1.0)
        if len(processed_image.shape) == 3:
            processed_image = cv2.GaussianBlur(processed_image, (kernel_size, kernel_size), sigma)
        else:
            processed_image = cv2.GaussianBlur(processed_image, (kernel_size, kernel_size), sigma)
    
    # Median filter for noise reduction
    if preprocessing_options.get('median_filter', False):
        kernel_size = preprocessing_options.get('median_kernel_size', 5)
        if len(processed_image.shape) == 3:
            processed_image = cv2.medianBlur(processed_image, kernel_size)
        else:
            processed_image = cv2.medianBlur(processed_image, kernel_size)
    
    # Morphological opening (erosion followed by dilation)
    if preprocessing_options.get('morphological_opening', False):
        kernel_size = preprocessing_options.get('morphological_kernel_size', 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if len(processed_image.shape) == 3:
            # Apply to each channel
            for i in range(processed_image.shape[2]):
                processed_image[:, :, i] = cv2.morphologyEx(processed_image[:, :, i], cv2.MORPH_OPEN, kernel)
        else:
            processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, kernel)
    
    # Morphological closing (dilation followed by erosion)
    if preprocessing_options.get('morphological_closing', False):
        kernel_size = preprocessing_options.get('morphological_kernel_size', 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if len(processed_image.shape) == 3:
            # Apply to each channel
            for i in range(processed_image.shape[2]):
                processed_image[:, :, i] = cv2.morphologyEx(processed_image[:, :, i], cv2.MORPH_CLOSE, kernel)
        else:
            processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)
    
    # Contrast enhancement using CLAHE
    if preprocessing_options.get('contrast_enhancement', False):
        clip_limit = preprocessing_options.get('clahe_clip_limit', 2.0)
        grid_size = preprocessing_options.get('clahe_grid_size', 8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        
        if len(processed_image.shape) == 3:
            # Convert to LAB color space and enhance L channel
            lab = cv2.cvtColor(processed_image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            processed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            processed_image = clahe.apply(processed_image)
    
    # Unsharp masking for edge enhancement
    if preprocessing_options.get('unsharp_mask', False):
        kernel_size = preprocessing_options.get('unsharp_kernel_size', 9)
        sigma = preprocessing_options.get('unsharp_sigma', 2.0)
        amount = preprocessing_options.get('unsharp_amount', 1.0)
        
        if len(processed_image.shape) == 3:
            # Apply to each channel
            for i in range(processed_image.shape[2]):
                blurred = cv2.GaussianBlur(processed_image[:, :, i], (kernel_size, kernel_size), sigma)
                processed_image[:, :, i] = cv2.addWeighted(processed_image[:, :, i], 1 + amount, blurred, -amount, 0)
        else:
            blurred = cv2.GaussianBlur(processed_image, (kernel_size, kernel_size), sigma)
            processed_image = cv2.addWeighted(processed_image, 1 + amount, blurred, -amount, 0)
    
    return processed_image


def calculate_area_statistics(masks: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate area statistics for a list of masks.
    
    Args:
        masks: List of mask dictionaries containing 'area' field
        
    Returns:
        Dictionary with area statistics including mean, std, median, etc.
    """
    if not masks:
        return {
            'mean': 0,
            'std': 0,
            'median': 0,
            'min': 0,
            'max': 0,
            'q25': 0,
            'q75': 0,
            'count': 0
        }
    
    areas = [mask['area'] for mask in masks]
    areas_array = np.array(areas)
    
    return {
        'mean': float(np.mean(areas_array)),
        'std': float(np.std(areas_array)),
        'median': float(np.median(areas_array)),
        'min': float(np.min(areas_array)),
        'max': float(np.max(areas_array)),
        'q25': float(np.percentile(areas_array, 25)),
        'q75': float(np.percentile(areas_array, 75)),
        'count': len(areas)
    }


def get_outlier_threshold(masks: List[Dict[str, Any]], method: str = 'std', factor: float = 2.0) -> Dict[str, float]:
    """
    Calculate outlier thresholds for mask areas.
    
    Args:
        masks: List of mask dictionaries
        method: Method for outlier detection ('std', 'iqr', 'percentile')
        factor: Multiplier for the threshold calculation
        
    Returns:
        Dictionary with 'min_area' and 'max_area' thresholds
    """
    if not masks:
        return {'min_area': 0, 'max_area': float('inf')}
    
    stats = calculate_area_statistics(masks)
    
    if method == 'std':
        # Use mean ± factor * std deviation
        min_threshold = max(0, stats['mean'] - factor * stats['std'])
        max_threshold = stats['mean'] + factor * stats['std']
    elif method == 'iqr':
        # Use IQR (Interquartile Range) method
        iqr = stats['q75'] - stats['q25']
        min_threshold = max(0, stats['q25'] - factor * iqr)
        max_threshold = stats['q75'] + factor * iqr
    elif method == 'percentile':
        # Use percentile-based thresholds
        lower_percentile = (100 - factor * 10) / 2  # e.g., factor=2 -> 5th percentile
        upper_percentile = 100 - lower_percentile   # e.g., factor=2 -> 95th percentile
        areas = [mask['area'] for mask in masks]
        min_threshold = float(np.percentile(areas, lower_percentile))
        max_threshold = float(np.percentile(areas, upper_percentile))
    else:
        # Default to std method
        min_threshold = max(0, stats['mean'] - factor * stats['std'])
        max_threshold = stats['mean'] + factor * stats['std']
    
    return {
        'min_area': min_threshold,
        'max_area': max_threshold
    }


def apply_preprocessing_filters(masks: List[Dict[str, Any]], image_shape: tuple) -> List[Dict[str, Any]]:
    """
    Apply automatic preprocessing filters during initial mask generation.
    
    This function removes:
    1. Masks that are >50% larger than the average mask area
    2. Masks that touch image edges (0 pixel distance)
    3. Masks with circularity < 0.85 (non-circular shapes)
    
    Args:
        masks: List of mask dictionaries from SAM
        image_shape: Shape of the original image (height, width, channels)
        
    Returns:
        List of preprocessed masks with outliers, edge-touching, and non-circular masks removed
    """
    if not masks or len(masks) < 3:
        return masks
    
    print(f"Preprocessing filters: Starting with {len(masks)} masks")
    
    # Step 1: Calculate average area
    areas = [mask['area'] for mask in masks]
    average_area = np.mean(areas)
    area_threshold = average_area * 1.5  # 50% larger than average
    
    print(f"Average mask area: {average_area:.1f} pixels")
    print(f"Area threshold (50% above average): {area_threshold:.1f} pixels")
    
    # Step 2: Apply preprocessing filters
    filtered_masks = []
    removed_large = 0
    removed_edge = 0
    removed_circularity = 0
    circularity_threshold = 0.6
    
    for mask in masks:
        area = mask['area']
        
        # Check if mask is too large (>50% larger than average)
        if area > area_threshold:
            removed_large += 1
            print(f"Removed large mask: area {area:.1f} > threshold {area_threshold:.1f}")
            continue
        
        # Check if mask touches edges (must have edge stats calculated)
        if 'edge_stats' in mask:
            if mask['edge_stats'].get('touches_edge', False):
                removed_edge += 1
                continue
        else:
            # Calculate edge stats if not present
            edge_stats = analyze_edge_proximity(mask['segmentation'], image_shape)
            mask['edge_stats'] = edge_stats
            if edge_stats.get('touches_edge', False):
                removed_edge += 1
                continue
        
        # Check circularity (must have pixel stats calculated)
        if 'pixel_stats' in mask and 'circularity' in mask['pixel_stats']:
            circularity = mask['pixel_stats']['circularity']
        else:
            # Calculate circularity if not present
            circularity = calculate_circularity(mask['segmentation'])
            if 'pixel_stats' not in mask:
                mask['pixel_stats'] = {}
            mask['pixel_stats']['circularity'] = circularity
        
        if circularity < circularity_threshold:
            removed_circularity += 1
            print(f"Removed non-circular mask: circularity {circularity:.3f} < threshold {circularity_threshold}")
            continue
        
        filtered_masks.append(mask)
    
    print(f"Preprocessing filters completed:")
    print(f"  - Removed {removed_large} masks for being >50% larger than average")
    print(f"  - Removed {removed_edge} masks for touching image edges")
    print(f"  - Removed {removed_circularity} masks for low circularity (< {circularity_threshold})")
    print(f"  - Remaining masks: {len(filtered_masks)}")
    
    return filtered_masks


def get_default_filters(image_shape: tuple, masks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Get default filter settings for user-defined filtering (after preprocessing).
    
    Args:
        image_shape: Shape of the image (height, width, channels)
        masks: Optional list of masks for reference (not used for thresholds anymore)
        
    Returns:
        Dictionary of default filter settings
    """
    height, width = image_shape[:2]
    
    # Base defaults - these are for user-defined filtering only
    # Preprocessing filters (area outliers and edge-touching) are applied automatically
    defaults = {
        'area_max': 15000,  # Additional maximum area limit if needed
        'area_min': 50,     # Minimum area to filter tiny artifacts
        # Note: edge-touching, statistical outlier removal, and circularity filtering are now done in preprocessing
        'enable_preprocessing_filters': True,  # Always enabled
        'preprocessing_area_factor': 1.5,      # 50% larger than average
        'preprocessing_remove_edge_touching': True,  # Always remove edge-touching
        'preprocessing_circularity_threshold': 0.85,  # Minimum circularity for circular objects
    }
    
    return defaults


def get_default_preprocessing() -> Dict[str, Any]:
    """
    Get default preprocessing settings for better segmentation.
    
    Returns:
        Dictionary of default preprocessing options
    """
    return {
        'gaussian_blur': True,
        'gaussian_kernel_size': 3,
        'gaussian_sigma': 1.0,
        'median_filter': False,
        'median_kernel_size': 3,
        'morphological_opening': False,
        'morphological_closing': False,
        'morphological_kernel_size': 3,
        'contrast_enhancement': True,
        'clahe_clip_limit': 2.0,
        'clahe_grid_size': 8,
        'unsharp_mask': False,
        'unsharp_kernel_size': 9,
        'unsharp_sigma': 2.0,
        'unsharp_amount': 1.0,
    }


def analyze_mask_pixels(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Analyze pixel intensities within a mask region.
    
    Args:
        image: Input image (RGB or grayscale)
        mask: Boolean mask array
        
    Returns:
        Dictionary containing pixel statistics
    """
    # Convert image to grayscale for intensity analysis
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    
    # Extract pixels within the mask
    mask_pixels = gray_image[mask]
    
    if len(mask_pixels) == 0:
        return {
            'mean_intensity': 0.0,
            'min_intensity': 0,
            'max_intensity': 0,
            'std_intensity': 0.0,
            'median_intensity': 0.0,
            'pixel_count': 0,
            'max_feret_diameter': 0.0,
            'circularity': 0.0
        }
    
    # Calculate maximum Feret diameter
    max_feret = calculate_max_feret_diameter(mask)
    
    # Calculate circularity
    circularity = calculate_circularity(mask)
    
    return {
        'mean_intensity': float(np.mean(mask_pixels)),
        'min_intensity': int(np.min(mask_pixels)),
        'max_intensity': int(np.max(mask_pixels)),
        'std_intensity': float(np.std(mask_pixels)),
        'median_intensity': float(np.median(mask_pixels)),
        'pixel_count': len(mask_pixels),
        'max_feret_diameter': max_feret,
        'circularity': circularity
    }


def analyze_edge_proximity(mask: np.ndarray, image_shape: tuple) -> Dict[str, Any]:
    """
    Analyze how close a mask is to the edges of the image.
    
    Args:
        mask: Boolean mask array
        image_shape: Shape of the image (height, width) or (height, width, channels)
        
    Returns:
        Dictionary containing edge proximity statistics
    """
    # Get image dimensions
    height, width = image_shape[:2]
    
    # Find the bounding box of the mask
    rows, cols = np.where(mask)
    
    if len(rows) == 0:
        # Empty mask
        return {
            'min_distance_to_edge': 0,
            'touches_edge': True,
            'distances': {'top': 0, 'bottom': 0, 'left': 0, 'right': 0},
            'edge_touching_sides': ['top', 'bottom', 'left', 'right']
        }
    
    # Calculate distances to each edge
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    distances = {
        'top': min_row,
        'bottom': height - 1 - max_row,
        'left': min_col, 
        'right': width - 1 - max_col
    }
    
    # Find minimum distance to any edge
    min_distance = min(distances.values())
    
    # Check which edges the mask is touching (distance = 0)
    edge_touching_sides = [edge for edge, dist in distances.items() if dist == 0]
    
    return {
        'min_distance_to_edge': min_distance,
        'touches_edge': len(edge_touching_sides) > 0,
        'distances': distances,
        'edge_touching_sides': edge_touching_sides
    }


def analyze_image_statistics(image: np.ndarray) -> Dict[str, Any]:
    """
    Analyze overall image statistics.
    
    Args:
        image: Input image (RGB or grayscale)
        
    Returns:
        Dictionary containing image statistics
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    
    # Calculate histogram
    histogram, _ = np.histogram(gray_image, bins=50, range=(0, 255))
    
    return {
        'overall_stats': {
            'mean_intensity': float(np.mean(gray_image)),
            'min_intensity': int(np.min(gray_image)),
            'max_intensity': int(np.max(gray_image)),
            'std_intensity': float(np.std(gray_image)),
            'median_intensity': float(np.median(gray_image)),
            'histogram': histogram.tolist()
        },
        'image_info': {
            'width': image.shape[1],
            'height': image.shape[0],
            'channels': image.shape[2] if len(image.shape) > 2 else 1,
            'total_pixels': gray_image.size
        }
    }


def filter_masks_by_criteria(
    masks: List[Dict[str, Any]], 
    image_shape: tuple,
    criteria: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Filter masks based on user-defined criteria.
    
    Note: Preprocessing filters (area outliers and edge-touching) are applied
    automatically before this function is called.
    
    Args:
        masks: List of mask dictionaries to filter (already preprocessed)
        image_shape: Shape of the original image (height, width, channels)
        criteria: Dictionary of filtering criteria
        
    Returns:
        List of filtered masks
    """
    if not masks:
        return masks
    
    if criteria is None:
        criteria = {}
    
    filtered_masks = []
    
    # Apply user-defined criteria only
    for mask in masks:
        mask_stats = mask.get('pixel_stats', {})
        edge_stats = mask.get('edge_stats', {})
        area = mask['area']
        
        # Check area bounds (user-defined limits)
        if 'area_min' in criteria and area < criteria['area_min']:
            continue
        if 'area_max' in criteria and area > criteria['area_max']:
            continue
            
        # Check mean intensity bounds
        if 'mean_min' in criteria and mask_stats.get('mean_intensity', 0) < criteria['mean_min']:
            continue
        if 'mean_max' in criteria and mask_stats.get('mean_intensity', 255) > criteria['mean_max']:
            continue
            
        # Check pixel intensity thresholds
        if 'min_threshold' in criteria and mask_stats.get('min_intensity', 0) < criteria['min_threshold']:
            continue
        if 'max_threshold' in criteria and mask_stats.get('max_intensity', 255) > criteria['max_threshold']:
            continue
            
        # Check standard deviation bounds (texture/variation)
        if 'std_min' in criteria and mask_stats.get('std_intensity', 0) < criteria['std_min']:
            continue
        if 'std_max' in criteria and mask_stats.get('std_intensity', float('inf')) > criteria['std_max']:
            continue
            
        # Check additional edge proximity (beyond preprocessing)
        if 'min_edge_distance' in criteria:
            min_dist = edge_stats.get('min_distance_to_edge', float('inf'))
            if min_dist < criteria['min_edge_distance']:
                continue
        
        filtered_masks.append(mask)
    
    return filtered_masks


def suggest_filter_values(image_stats: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Suggest filter values based on image analysis.
    
    Args:
        image_stats: Image statistics from analyze_image_statistics
        
    Returns:
        Dictionary of suggested filter ranges
    """
    overall = image_stats['overall_stats']
    image_info = image_stats['image_info']
    mean = overall['mean_intensity']
    std = overall['std_intensity']
    
    # Calculate suggested edge distance based on image size
    min_dimension = min(image_info['width'], image_info['height'])
    suggested_edge_distance = max(10, min_dimension // 20)  # 5% of smallest dimension, minimum 10 pixels
    
    return {
        'bright_objects': {
            'mean_min': mean + std * 0.5,
            'description': 'Objects brighter than average'
        },
        'dark_objects': {
            'mean_max': mean - std * 0.5,
            'description': 'Objects darker than average'
        },
        'high_contrast': {
            'std_min': std * 1.5,
            'description': 'Objects with high internal variation'
        },
        'uniform_objects': {
            'std_max': std * 0.5,
            'description': 'Objects with uniform intensity'
        },
        'edge_filtered': {
            'min_edge_distance': suggested_edge_distance,
            'description': f'Objects at least {suggested_edge_distance} pixels from edges'
        },
        'exclude_edge_touching': {
            'exclude_edge_touching': 1,  # Boolean value as float
            'description': 'Exclude objects touching image edges'
        }
    }


class FilterPresets:
    """Common filter presets for different use cases."""
    
    @staticmethod
    def bright_droplets(image_stats: Dict) -> Dict[str, float]:
        """Filter for bright droplets/particles."""
        mean = image_stats['overall_stats']['mean_intensity']
        return {
            'mean_min': mean * 1.2,
            'area_min': 50,
            'area_max': 10000
        }
    
    @staticmethod
    def dark_droplets(image_stats: Dict) -> Dict[str, float]:
        """Filter for dark droplets/particles."""
        mean = image_stats['overall_stats']['mean_intensity']
        return {
            'mean_max': mean * 0.8,
            'area_min': 50,
            'area_max': 10000
        }
    
    @staticmethod
    def large_objects(image_stats: Dict) -> Dict[str, float]:
        """Filter for large objects."""
        total_pixels = image_stats['image_info']['total_pixels']
        return {
            'area_min': total_pixels // 1000,  # At least 0.1% of image
            'area_max': total_pixels // 10     # At most 10% of image
        }
    
    @staticmethod
    def small_objects(image_stats: Dict) -> Dict[str, float]:
        """Filter for small objects."""
        return {
            'area_min': 10,
            'area_max': 1000
        }
    
    @staticmethod
    def complete_objects_only(image_stats: Dict) -> Dict[str, float]:
        """Filter for complete objects that don't touch image edges."""
        image_info = image_stats['image_info']
        min_dimension = min(image_info['width'], image_info['height'])
        edge_buffer = max(5, min_dimension // 50)  # 2% of smallest dimension, minimum 5 pixels
        
        return {
            'exclude_edge_touching': 1,  # Boolean value as float
            'min_edge_distance': edge_buffer,
            'area_min': 50
        }
    
    @staticmethod
    def center_objects(image_stats: Dict) -> Dict[str, float]:
        """Filter for objects in the center region of the image."""
        image_info = image_stats['image_info']
        min_dimension = min(image_info['width'], image_info['height'])
        center_buffer = min_dimension // 4  # Objects must be at least 25% of image dimension from edges
        
        return {
            'min_edge_distance': center_buffer,
            'area_min': 100
        }


def calculate_summary_statistics(masks: List[Dict[str, Any]], intensity_threshold: float = None) -> Dict[str, Any]:
    """
    Calculate summary statistics for masks including diameter and intensity analysis.
    
    Args:
        masks: List of mask dictionaries with pixel_stats
        intensity_threshold: Threshold to separate high/low intensity groups
        
    Returns:
        Dictionary with summary statistics
    """
    if not masks:
        return {
            'total_masks': 0,
            'average_diameter': 0,
            'diameter_stats': {},
            'intensity_analysis': {},
            'classification_enabled': False
        }
    
    # Extract diameter data from masks that have it
    diameters = []
    intensities = []
    
    for mask in masks:
        if 'pixel_stats' in mask:
            pixel_stats = mask['pixel_stats']
            
            # Collect diameter data
            if 'max_feret_diameter' in pixel_stats and pixel_stats['max_feret_diameter'] > 0:
                diameters.append(pixel_stats['max_feret_diameter'])
            
            # Collect intensity data
            if 'mean_intensity' in pixel_stats:
                intensities.append(pixel_stats['mean_intensity'])
    
    # Calculate diameter statistics
    diameter_stats = {}
    if diameters:
        diameter_array = np.array(diameters)
        mean_diameter = float(np.mean(diameter_array))
        std_diameter = float(np.std(diameter_array))
        cv_diameter = (std_diameter / mean_diameter * 100) if mean_diameter > 0 else 0
        diameter_stats = {
            'count': len(diameters),
            'mean': mean_diameter,
            'std': std_diameter,
            'cv': cv_diameter,
            'median': float(np.median(diameter_array)),
            'min': float(np.min(diameter_array)),
            'max': float(np.max(diameter_array))
        }
    
    # Calculate intensity analysis if threshold is provided
    intensity_analysis = {}
    classification_enabled = intensity_threshold is not None
    
    if classification_enabled and intensities:
        intensities_array = np.array(intensities)
        
        # Split into high and low intensity groups
        high_intensity_masks = intensities_array[intensities_array >= intensity_threshold]
        low_intensity_masks = intensities_array[intensities_array < intensity_threshold]
        
        # Calculate statistics for high intensity group
        high_intensity_stats = {}
        if len(high_intensity_masks) > 0:
            high_intensity_stats = {
                'count': len(high_intensity_masks),
                'mean': float(np.mean(high_intensity_masks)),
                'std': float(np.std(high_intensity_masks)),
                'min': float(np.min(high_intensity_masks)),
                'max': float(np.max(high_intensity_masks))
            }
        
        # Calculate statistics for low intensity group
        low_intensity_stats = {}
        if len(low_intensity_masks) > 0:
            low_intensity_stats = {
                'count': len(low_intensity_masks),
                'mean': float(np.mean(low_intensity_masks)),
                'std': float(np.std(low_intensity_masks)),
                'min': float(np.min(low_intensity_masks)),
                'max': float(np.max(low_intensity_masks))
            }
        
        # Overall intensity statistics
        overall_intensity_stats = {
            'count': len(intensities_array),
            'mean': float(np.mean(intensities_array)),
            'std': float(np.std(intensities_array)),
            'threshold': intensity_threshold
        }
        
        intensity_analysis = {
            'threshold': intensity_threshold,
            'overall': overall_intensity_stats,
            'high_intensity': high_intensity_stats,
            'low_intensity': low_intensity_stats
        }
    
    return {
        'total_masks': len(masks),
        'masks_with_diameter': len(diameters),
        'masks_with_intensity': len(intensities),
        'average_diameter': diameter_stats.get('mean', 0),
        'diameter_stats': diameter_stats,
        'intensity_analysis': intensity_analysis,
        'classification_enabled': classification_enabled
    } 