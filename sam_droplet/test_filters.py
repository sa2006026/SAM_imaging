#!/usr/bin/env python3
"""
Test script for the SAM droplet segmentation filtering system.
Demonstrates the various filtering capabilities including edge proximity.
"""

import numpy as np
import cv2
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from sam_droplet.filters import (
    analyze_mask_pixels,
    analyze_image_statistics,
    analyze_edge_proximity,
    filter_masks_by_criteria,
    suggest_filter_values,
    FilterPresets
)


def create_test_image_and_masks():
    """Create a test image with various objects for testing filters."""
    # Create a 300x300 test image
    image = np.ones((300, 300, 3), dtype=np.uint8) * 128  # Gray background
    
    # Add some bright objects
    cv2.circle(image, (50, 50), 20, (200, 200, 200), -1)      # Bright, near top-left edge
    cv2.circle(image, (150, 150), 30, (180, 180, 180), -1)    # Medium bright, center
    cv2.circle(image, (250, 250), 15, (220, 220, 220), -1)    # Bright, near bottom-right edge
    cv2.circle(image, (75, 200), 25, (80, 80, 80), -1)        # Dark, away from edges
    cv2.circle(image, (5, 150), 18, (160, 160, 160), -1)      # Medium, touching left edge
    
    # Create corresponding masks
    masks = []
    
    # Mask 1: Bright object near edge (top-left)
    mask1 = np.zeros((300, 300), dtype=bool)
    cv2.circle(mask1.astype(np.uint8), (50, 50), 20, 1, -1)
    masks.append({
        'segmentation': mask1,
        'area': np.sum(mask1),
        'bbox': [30, 30, 40, 40],
        'stability_score': 0.95
    })
    
    # Mask 2: Medium bright object in center
    mask2 = np.zeros((300, 300), dtype=bool)
    cv2.circle(mask2.astype(np.uint8), (150, 150), 30, 1, -1)
    masks.append({
        'segmentation': mask2,
        'area': np.sum(mask2),
        'bbox': [120, 120, 60, 60],
        'stability_score': 0.92
    })
    
    # Mask 3: Bright object near edge (bottom-right)
    mask3 = np.zeros((300, 300), dtype=bool)
    cv2.circle(mask3.astype(np.uint8), (250, 250), 15, 1, -1)
    masks.append({
        'segmentation': mask3,
        'area': np.sum(mask3),
        'bbox': [235, 235, 30, 30],
        'stability_score': 0.88
    })
    
    # Mask 4: Dark object away from edges
    mask4 = np.zeros((300, 300), dtype=bool)
    cv2.circle(mask4.astype(np.uint8), (75, 200), 25, 1, -1)
    masks.append({
        'segmentation': mask4,
        'area': np.sum(mask4),
        'bbox': [50, 175, 50, 50],
        'stability_score': 0.85
    })
    
    # Mask 5: Object touching edge
    mask5 = np.zeros((300, 300), dtype=bool)
    cv2.circle(mask5.astype(np.uint8), (5, 150), 18, 1, -1)
    masks.append({
        'segmentation': mask5,
        'area': np.sum(mask5),
        'bbox': [0, 132, 23, 36],
        'stability_score': 0.75
    })
    
    return image, masks


def test_edge_proximity_analysis():
    """Test the edge proximity analysis function."""
    print("🔍 Testing Edge Proximity Analysis")
    print("=" * 50)
    
    image, masks = create_test_image_and_masks()
    
    for i, mask in enumerate(masks):
        edge_stats = analyze_edge_proximity(mask['segmentation'], image.shape)
        print(f"\nMask {i+1}:")
        print(f"  Min distance to edge: {edge_stats['min_distance_to_edge']} pixels")
        print(f"  Touches edge: {edge_stats['touches_edge']}")
        print(f"  Distances: {edge_stats['distances']}")
        if edge_stats['edge_touching_sides']:
            print(f"  Touching sides: {edge_stats['edge_touching_sides']}")
    
    print("\n" + "=" * 50)


def test_edge_filtering():
    """Test edge proximity filtering."""
    print("\n🚫 Testing Edge Proximity Filtering")
    print("=" * 50)
    
    image, masks = create_test_image_and_masks()
    
    print(f"Original masks: {len(masks)}")
    
    # Test 1: Exclude edge-touching objects
    print("\nTest 1: Exclude edge-touching objects")
    filter1 = {'exclude_edge_touching': True}
    filtered1 = filter_masks_by_criteria(masks, image, filter1)
    print(f"After excluding edge-touching: {len(filtered1)} masks")
    
    # Test 2: Minimum distance from edge
    print("\nTest 2: Minimum 30 pixels from edge")
    filter2 = {'min_edge_distance': 30}
    filtered2 = filter_masks_by_criteria(masks, image, filter2)
    print(f"After min distance filter: {len(filtered2)} masks")
    
    # Test 3: Combined edge and brightness filtering
    print("\nTest 3: Combined edge + brightness filtering")
    filter3 = {
        'exclude_edge_touching': True,
        'min_edge_distance': 10,
        'mean_min': 150
    }
    filtered3 = filter_masks_by_criteria(masks, image, filter3)
    print(f"After combined filtering: {len(filtered3)} masks")
    
    # Show details of remaining masks
    if filtered3:
        print("\nRemaining masks details:")
        for i, mask in enumerate(filtered3):
            pixel_stats = mask['pixel_stats']
            edge_stats = mask['edge_stats']
            print(f"  Mask {i+1}: Mean={pixel_stats['mean_intensity']:.1f}, "
                  f"Edge distance={edge_stats['min_distance_to_edge']}")
    
    print("\n" + "=" * 50)


def test_filter_presets():
    """Test the new filter presets that include edge filtering."""
    print("\n🎯 Testing New Filter Presets")
    print("=" * 50)
    
    image, masks = create_test_image_and_masks()
    image_stats = analyze_image_statistics(image)
    
    # Test complete objects preset
    print("Testing 'complete_objects_only' preset:")
    complete_filter = FilterPresets.complete_objects_only(image_stats)
    print(f"Filter: {complete_filter}")
    filtered_complete = filter_masks_by_criteria(masks, image, complete_filter)
    print(f"Results: {len(filtered_complete)} masks (from {len(masks)} original)")
    
    # Test center objects preset
    print("\nTesting 'center_objects' preset:")
    center_filter = FilterPresets.center_objects(image_stats)
    print(f"Filter: {center_filter}")
    filtered_center = filter_masks_by_criteria(masks, image, center_filter)
    print(f"Results: {len(filtered_center)} masks (from {len(masks)} original)")
    
    print("\n" + "=" * 50)


def demo_filtering():
    """Demonstrate the filtering functionality."""
    print("🔬 SAM Droplet Segmentation - Pixel Filtering Demo")
    print("=" * 50)
    
    print("\n📋 Available Filter Types:")
    print("• Mean Intensity Filters (mean_min, mean_max)")
    print("  - Filter objects based on average brightness")
    print("  - Useful for separating bright vs dark objects")
    
    print("\n• Pixel Threshold Filters (min_threshold, max_threshold)")
    print("  - Filter based on darkest/brightest pixels in objects")
    print("  - Good for removing objects with extreme pixel values")
    
    print("\n• Texture/Variation Filters (std_min, std_max)")
    print("  - Filter based on pixel intensity variation")
    print("  - std_min: Objects with high internal contrast")
    print("  - std_max: Objects with uniform intensity")
    
    print("\n• Area Filters (area_min, area_max)")
    print("  - Filter based on object size in pixels")
    print("  - Remove objects that are too small or too large")
    
    print("\n🎯 Example Filter Scenarios:")
    print("-" * 30)
    
    print("\n1. Bright droplets only:")
    print("   {'mean_min': 150, 'area_min': 50, 'area_max': 5000}")
    
    print("\n2. Dark objects with high contrast:")
    print("   {'mean_max': 100, 'std_min': 15}")
    
    print("\n3. Large, uniform objects:")
    print("   {'area_min': 1000, 'std_max': 10}")
    
    print("\n4. Small, textured particles:")
    print("   {'area_max': 500, 'std_min': 20}")
    
    print("\n🚀 Usage Instructions:")
    print("-" * 20)
    print("1. Start the Flask server: python app.py")
    print("2. Open browser to http://localhost:9487")
    print("3. Upload an image")
    print("4. Click 'Analyze Image' to see image statistics")
    print("5. Set filter values based on analysis")
    print("6. Click 'Apply Filters' to segment with filtering")
    print("7. Or click 'Generate Masks' for unfiltered segmentation")
    
    print("\n📊 Filter Value Guidelines:")
    print("-" * 25)
    print("• Intensity values: 0-255 (0=black, 255=white)")
    print("• Mean intensity: Average brightness of object pixels")
    print("• Std deviation: Higher = more texture/variation")
    print("• Area: Number of pixels in the object")
    
    print("\n✨ Tips:")
    print("• Use 'Analyze Image' to understand your image statistics")
    print("• Start with broader filters, then refine")
    print("• Combine multiple filter types for precise selection")
    print("• Check pixel statistics in hover preview to tune filters")

if __name__ == "__main__":
    demo_filtering()
    test_edge_proximity_analysis()
    test_edge_filtering()
    test_filter_presets() 