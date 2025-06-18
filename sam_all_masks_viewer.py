import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import json
import time

def download_sam_model(model_type="vit_h"):
    """Download SAM model if not present"""
    import urllib.request
    
    model_urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
    
    model_path = f"sam_{model_type}_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Downloading SAM {model_type} model...")
        urllib.request.urlretrieve(model_urls[model_type], model_path)
        print("Model downloaded successfully!")
    
    return model_path

def load_image(image_path):
    """Load and convert image to RGB"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def generate_masks(image, model_type="vit_h", crop_n_layers=3):
    """Generate masks using SAM with specified model and crop layers"""
    print(f"Using SAM {model_type} model with crop_n_layers={crop_n_layers}")
    
    # Clear GPU cache first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Download model if needed
    model_path = download_sam_model(model_type)
    
    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    
    # Create mask generator with memory-optimized settings
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,  # Reduced from 64 to save memory
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=crop_n_layers
    )
    
    # Generate masks
    print("Generating masks...")
    start_time = time.time()
    
    try:
        masks = mask_generator.generate(image)
        end_time = time.time()
        print(f"Generated {len(masks)} masks in {end_time - start_time:.2f} seconds")
        
        # Clear GPU cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return masks
        
    except torch.cuda.OutOfMemoryError:
        print("GPU out of memory! Trying with reduced settings...")
        
        # Clear cache and try with even lower settings
        torch.cuda.empty_cache()
        del sam, mask_generator
        torch.cuda.empty_cache()
        
        # Reload with more conservative settings
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)
        
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,  # Even lower resolution
            pred_iou_thresh=0.9,
            stability_score_thresh=0.95,
            crop_n_layers=2,  # Reduce crop layers
            crop_n_points_downscale_factor=4,  # Increase downscale
            min_mask_region_area=200,  # Larger minimum area
        )
        
        print("Retrying with reduced settings: points_per_side=16, crop_n_layers=2")
        masks = mask_generator.generate(image)
        end_time = time.time()
        print(f"Generated {len(masks)} masks in {end_time - start_time:.2f} seconds")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return masks

def create_all_masks_visualization(image, masks, output_path, crop_n_layers=3, points_per_side=32, show_bboxes=True, show_labels=False):
    """Create a comprehensive visualization showing all masks in one image"""
    print(f"Creating comprehensive visualization with {len(masks)} masks...")
    
    # Create figure with high resolution
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    
    # Show original image as background
    ax.imshow(image)
    
    # Generate distinct colors for all masks
    colors = plt.cm.tab20(np.linspace(0, 1, 20))  # 20 distinct colors
    extended_colors = []
    
    # Extend colors to cover all masks
    color_cycles = (len(masks) // 20) + 1
    for _ in range(color_cycles):
        extended_colors.extend(colors)
    
    # Create a single overlay for all masks
    overlay = np.zeros((*image.shape[:2], 4))  # RGBA
    
    # Sort masks by area (largest first) for better visualization, but keep original indices
    masks_with_indices = [(i, mask) for i, mask in enumerate(masks)]
    sorted_masks_with_indices = sorted(masks_with_indices, key=lambda x: x[1]['area'], reverse=True)
    
    # Process masks in batches for better performance
    batch_size = 50
    mask_count = 0
    
    for batch_start in range(0, len(sorted_masks_with_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(sorted_masks_with_indices))
        
        for i in range(batch_start, batch_end):
            original_index, mask_data = sorted_masks_with_indices[i]
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']  # [x, y, width, height]
            area = mask_data['area']
            stability_score = mask_data['stability_score']
            
            # Use color based on original index
            color = extended_colors[mask_count % len(extended_colors)]
            
            # Add mask to overlay with transparency based on size
            # Larger masks get lower opacity to not overwhelm smaller ones
            alpha = max(0.3, min(0.7, 1.0 - (area / max(m['area'] for m in masks))))
            overlay[mask] = [color[0], color[1], color[2], alpha]
            
            if show_bboxes:
                # Draw bounding box
                x, y, w, h = bbox
                rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                       edgecolor=color[:3], facecolor='none', alpha=0.8)
                ax.add_patch(rect)
                
                if show_labels and mask_count < 50:  # Show labels for first 50 masks only
                    # Add mask info text - use original index to match metadata
                    label = f'{original_index}\nA:{area}'
                    ax.text(x, y-10, label, fontsize=6, color=color[:3], 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.3),
                           verticalalignment='bottom')
            
            mask_count += 1
    
    # Apply the overlay
    ax.imshow(overlay)
    
    # Set title with comprehensive information
    title = f'SAM Segmentation Results\n'
    title += f'Model: vit_h, Crop Layers: {crop_n_layers}, Points per Side: {points_per_side}\n'
    title += f'Total Masks: {len(masks)}, Image Size: {image.shape[1]}×{image.shape[0]}'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved comprehensive visualization: {output_path}")

def create_mask_statistics_plot(masks, output_path):
    """Create statistical plots about the masks"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract statistics
    areas = [mask['area'] for mask in masks]
    stability_scores = [mask['stability_score'] for mask in masks]
    bbox_widths = [mask['bbox'][2] for mask in masks]
    bbox_heights = [mask['bbox'][3] for mask in masks]
    
    # 1. Area distribution
    axes[0, 0].hist(areas, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Mask Area (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Mask Areas')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Stability score distribution
    axes[0, 1].hist(stability_scores, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Stability Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Stability Scores')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Bounding box dimensions
    axes[1, 0].scatter(bbox_widths, bbox_heights, alpha=0.6, s=10)
    axes[1, 0].set_xlabel('Bounding Box Width')
    axes[1, 0].set_ylabel('Bounding Box Height')
    axes[1, 0].set_title('Bounding Box Dimensions')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Area vs Stability Score
    axes[1, 1].scatter(areas, stability_scores, alpha=0.6, s=10)
    axes[1, 1].set_xlabel('Mask Area (pixels)')
    axes[1, 1].set_ylabel('Stability Score')
    axes[1, 1].set_title('Area vs Stability Score')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved statistics plot: {output_path}")

def save_mask_metadata(masks, output_path):
    """Save detailed metadata about all masks"""
    metadata = {
        'total_masks': len(masks),
        'model_config': {
            'model_type': 'vit_h',
            'crop_n_layers': 3,
            'points_per_side': 64
        },
        'statistics': {
            'min_area': int(min(mask['area'] for mask in masks)),
            'max_area': int(max(mask['area'] for mask in masks)),
            'mean_area': float(np.mean([mask['area'] for mask in masks])),
            'median_area': float(np.median([mask['area'] for mask in masks])),
            'std_area': float(np.std([mask['area'] for mask in masks])),
            'min_stability': float(min(mask['stability_score'] for mask in masks)),
            'max_stability': float(max(mask['stability_score'] for mask in masks)),
            'mean_stability': float(np.mean([mask['stability_score'] for mask in masks]))
        },
        'mask_details': []
    }
    
    # Add details for each mask
    for i, mask in enumerate(masks):
        mask_detail = {
            'id': i,
            'area': int(mask['area']),
            'bbox': [int(x) for x in mask['bbox']],
            'stability_score': float(mask['stability_score']),
            'predicted_iou': float(mask['predicted_iou'])
        }
        metadata['mask_details'].append(mask_detail)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata: {output_path}")

def main():
    # Configuration
    model_type = "vit_h"
    crop_n_layers = 3
    points_per_side = 32  # Track the points per side used
    
    # Paths
    image_path = "image/Test/GY_image.png"
    base_output_dir = "image/Output"
    
    # Create output directory
    output_folder_name = f"all_masks_{model_type}_layer_{crop_n_layers}"
    output_dir = os.path.join(base_output_dir, output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Loading image: {image_path}")
    print(f"Output directory: {output_dir}")
    
    # Load image
    image = load_image(image_path)
    print(f"Image shape: {image.shape}")
    
    # Generate masks using SAM huge model
    masks = generate_masks(image, model_type=model_type, crop_n_layers=crop_n_layers)
    
    print("\nCreating visualizations...")
    
    # Create comprehensive all-masks visualization
    create_all_masks_visualization(
        image, masks,
        os.path.join(output_dir, "all_masks_comprehensive.png"),
        crop_n_layers=crop_n_layers,
        points_per_side=points_per_side,
        show_bboxes=True,
        show_labels=False
    )
    
    # Create version with labels (for smaller subset)
    create_all_masks_visualization(
        image, masks[:100],  # Show labels for first 100 masks only
        os.path.join(output_dir, "all_masks_with_labels.png"),
        crop_n_layers=crop_n_layers,
        points_per_side=points_per_side,
        show_bboxes=True,
        show_labels=True
    )
    
    # Create statistics plot
    create_mask_statistics_plot(masks, os.path.join(output_dir, "mask_statistics.png"))
    
    # Save metadata
    save_mask_metadata(masks, os.path.join(output_dir, "all_masks_metadata.json"))
    
    print("\n" + "="*70)
    print("SEGMENTATION COMPLETE!")
    print(f"Model: {model_type}")
    print(f"Crop layers: {crop_n_layers}")
    print(f"Points per side: {points_per_side}")
    print(f"Output directory: {output_dir}")
    print(f"Total masks generated: {len(masks)}")
    print(f"Image dimensions: {image.shape[1]}×{image.shape[0]} pixels")
    
    # Area statistics
    areas = [mask['area'] for mask in masks]
    print(f"\nMask area statistics:")
    print(f"  Smallest mask: {min(areas):,} pixels")
    print(f"  Largest mask: {max(areas):,} pixels")
    print(f"  Average mask: {np.mean(areas):,.0f} pixels")
    print(f"  Median mask: {np.median(areas):,.0f} pixels")
    
    print("\nOutput files created:")
    print("- all_masks_comprehensive.png (all masks with bounding boxes)")
    print("- all_masks_with_labels.png (first 100 masks with labels)")
    print("- mask_statistics.png (statistical analysis)")
    print("- all_masks_metadata.json (detailed metadata)")
    print("="*70)

if __name__ == "__main__":
    main() 