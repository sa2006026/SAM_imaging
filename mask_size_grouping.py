import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from sklearn.cluster import KMeans
import json

def download_sam_model(model_type="vit_h"):
    """Download SAM model if not present"""
    import urllib.request
    
    model_urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
    
    model_path = f"model/sam_{model_type}_model.pth"
    
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
    # Download model if needed
    model_path = download_sam_model(model_type)
    
    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using model: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    
    # Create mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        crop_n_layers=crop_n_layers,

    )
    
    # Generate masks
    print("Generating masks...")
    masks = mask_generator.generate(image)
    print(f"Generated {len(masks)} masks")
    
    return masks

def calculate_circularity(mask):
    """Calculate circularity of a mask (4π*area/perimeter²)"""
    import cv2
    
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate area and perimeter
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    if perimeter == 0:
        return 0.0
    
    # Circularity = 4π * area / perimeter²
    # Perfect circle = 1.0, lower values = less circular
    circularity = (4 * np.pi * area) / (perimeter * perimeter)
    return min(circularity, 1.0)  # Cap at 1.0

def add_circularity_to_masks(masks):
    """Add circularity information to all masks"""
    print("Calculating circularity for all masks...")
    
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        circularity = calculate_circularity(mask)
        mask_data['circularity'] = circularity
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(masks)} masks...")
    
    print(f"Circularity calculation complete for {len(masks)} masks")
    return masks

def filter_by_circularity(masks, min_circularity):
    """Filter masks by circularity threshold"""
    print(f"Filtering masks by circularity (min_circularity: {min_circularity})...")
    
    original_count = len(masks)
    filtered_masks = []
    removed_low_circularity = []
    
    for mask_data in masks:
        circularity = mask_data.get('circularity', 0.0)
        
        if circularity >= min_circularity:
            filtered_masks.append(mask_data)
        else:
            removed_low_circularity.append(mask_data)
    
    print(f"Circularity filtering results:")
    print(f"  Original masks: {original_count}")
    print(f"  Removed low circularity masks: {len(removed_low_circularity)}")
    print(f"  Remaining masks: {len(filtered_masks)}")
    
    return filtered_masks, removed_low_circularity

def calculate_blob_distance(mask):
    """Calculate maximum distance between connected components (blobs) in a mask"""
    import cv2
    
    # Convert mask to uint8
    mask_uint8 = mask.astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask_uint8, connectivity=8)
    
    # If only one blob (plus background), return 0
    if num_labels <= 2:  # background + 1 blob
        return 0.0
    
    # Calculate centroids of each blob
    centroids = []
    for label in range(1, num_labels):  # Skip background (label 0)
        blob_pixels = np.where(labels == label)
        if len(blob_pixels[0]) > 0:
            centroid_y = np.mean(blob_pixels[0])
            centroid_x = np.mean(blob_pixels[1])
            centroids.append((centroid_x, centroid_y))
    
    # Calculate maximum distance between any two centroids
    max_distance = 0.0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distance = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                             (centroids[i][1] - centroids[j][1])**2)
            max_distance = max(max_distance, distance)
    
    return max_distance

def filter_by_blob_distance(masks, max_distance=50):
    """Filter out masks where blobs are separated by more than max_distance pixels"""
    print(f"Filtering masks by blob distance (max_distance: {max_distance} pixels)...")
    
    original_count = len(masks)
    filtered_masks = []
    removed_multi_blob = []
    
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        blob_distance = calculate_blob_distance(mask)
        
        # Add blob distance to mask metadata for analysis
        mask_data['blob_distance'] = blob_distance
        
        if blob_distance <= max_distance:
            filtered_masks.append(mask_data)
        else:
            removed_multi_blob.append(mask_data)
        
        if i % 100 == 0:
            print(f"Processed {i}/{original_count} masks...")
    
    print(f"Blob distance filtering results:")
    print(f"  Original masks: {original_count}")
    print(f"  Removed distant-blob masks: {len(removed_multi_blob)}")
    print(f"  Remaining masks: {len(filtered_masks)}")
    
    # Show some statistics about blob distances
    if removed_multi_blob:
        distances = [mask['blob_distance'] for mask in removed_multi_blob]
        print(f"  Removed mask distances - min: {min(distances):.1f}, max: {max(distances):.1f}, mean: {np.mean(distances):.1f}")
    
    if filtered_masks:
        distances = [mask['blob_distance'] for mask in filtered_masks]
        print(f"  Kept mask distances - min: {min(distances):.1f}, max: {max(distances):.1f}, mean: {np.mean(distances):.1f}")
    
    return filtered_masks, removed_multi_blob

def extract_mask_features(masks):
    """Extract features for K-means clustering including circularity"""
    features = []
    
    for mask_data in masks:
        mask = mask_data['segmentation']
        bbox = mask_data['bbox']
        
        # Feature vector: [area, bbox_width, bbox_height, aspect_ratio, stability_score, circularity]
        area = mask_data['area']
        width = bbox[2]
        height = bbox[3]
        aspect_ratio = width / height if height > 0 else 1.0
        stability_score = mask_data['stability_score']
        circularity = mask_data.get('circularity', 0.0)
        
        features.append([
            area,
            width,
            height,
            aspect_ratio,
            stability_score,
            circularity
        ])
    
    return np.array(features)

def cluster_masks_kmeans(masks, n_clusters=2):
    """Cluster masks into groups using K-means with improved preprocessing"""
    print(f"Clustering masks into {n_clusters} groups using K-means...")
    
    # Extract features
    features = extract_mask_features(masks)
    
    # Apply log transformation to area to handle extreme values
    features_processed = features.copy()
    features_processed[:, 0] = np.log1p(features_processed[:, 0])  # log(area + 1)
    
    # Normalize features for better clustering
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_processed)
    
    # Always use K-means clustering based on features, not forced splits
    # Let K-means naturally discover clusters based on feature similarity
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_normalized)
    
    # Check if clustering is too imbalanced and try different approaches
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    min_cluster_size = min(counts)
    max_cluster_size = max(counts)
    imbalance_ratio = max_cluster_size / min_cluster_size
    
    print(f"Initial clustering balance: {counts}")
    
    # If clustering is extremely imbalanced (ratio > 10), try alternative approaches
    if imbalance_ratio > 10:
        print(f"Clustering is imbalanced (ratio: {imbalance_ratio:.1f}), trying alternative methods...")
        
        # Try with different number of clusters first
        for n_alt in [3, 4, 5]:
            kmeans_alt = KMeans(n_clusters=n_alt, random_state=42, n_init=10)
            labels_alt = kmeans_alt.fit_predict(features_normalized)
            unique_alt, counts_alt = np.unique(labels_alt, return_counts=True)
            alt_ratio = max(counts_alt) / min(counts_alt)
            
            if alt_ratio < imbalance_ratio:
                print(f"Better clustering found with {n_alt} clusters (ratio: {alt_ratio:.1f})")
                # Merge smaller clusters into 2 main groups
                if n_alt > 2:
                    # Group smaller clusters together
                    sorted_indices = np.argsort(counts_alt)
                    large_cluster = sorted_indices[-1]  # Largest cluster
                    small_clusters = sorted_indices[:-1]  # All other clusters
                    
                    # Reassign labels to create 2 groups
                    new_labels = np.zeros_like(labels_alt)
                    new_labels[labels_alt == large_cluster] = 0  # Large objects
                    for small_idx in small_clusters:
                        new_labels[labels_alt == small_idx] = 1  # Small objects
                    
                    cluster_labels = new_labels
                    break
        
        # If still imbalanced, fall back to area-based natural clustering
        if imbalance_ratio > 10:
            areas = [mask['area'] for mask in masks]
            # Use natural breaks in area distribution instead of median
            area_sorted = np.sort(areas)
            
            # Find the biggest gap in area distribution
            gaps = np.diff(area_sorted)
            max_gap_idx = np.argmax(gaps)
            threshold = (area_sorted[max_gap_idx] + area_sorted[max_gap_idx + 1]) / 2
            
            print(f"Using natural area threshold: {threshold:.0f} (found largest gap in distribution)")
            cluster_labels = np.array([1 if area > threshold else 0 for area in areas])
    
    # Group masks by cluster
    clustered_masks = [[] for _ in range(n_clusters)]
    for i, label in enumerate(cluster_labels):
        clustered_masks[label].append(masks[i])
    
    # Print cluster statistics
    for i, cluster in enumerate(clustered_masks):
        if len(cluster) > 0:
            areas = [mask['area'] for mask in cluster]
            print(f"Cluster {i}: {len(cluster)} masks, "
                  f"area range: {min(areas):.0f}-{max(areas):.0f}, "
                  f"mean area: {np.mean(areas):.0f}")
        else:
            print(f"Cluster {i}: 0 masks")
    
    return clustered_masks, cluster_labels

def create_optimized_visualization(image, masks, title, output_path, colors=None, show_numbers=True, max_numbers=100, original_indices=None):
    """Create optimized visualization with mask numbers and better performance"""
    print(f"Creating visualization: {title}")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))  # Larger figure for better readability
    
    # Show original image
    ax.imshow(image)
    
    # Generate colors if not provided
    if colors is None:
        colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
    
    # Create a single overlay image instead of multiple overlays
    overlay = np.zeros((*image.shape[:2], 4))  # RGBA
    
    # Process masks in batches for better performance
    batch_size = 50
    for batch_start in range(0, len(masks), batch_size):
        batch_end = min(batch_start + batch_size, len(masks))
        
        for i in range(batch_start, batch_end):
            mask_data = masks[i]
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']  # [x, y, width, height]
            color = colors[i % len(colors)]
            
            # Add to overlay
            overlay[mask] = [color[0], color[1], color[2], 0.5]
            
            # Draw bounding box
            x, y, w, h = bbox
            rect = patches.Rectangle((x, y), w, h, linewidth=1.5, 
                                   edgecolor=color[:3], facecolor='none')
            ax.add_patch(rect)
            
            # Add mask ID number above the bounding box
            if show_numbers and i < max_numbers:
                # Use original index if provided, otherwise use current index
                display_id = original_indices[i] if original_indices is not None else i
                ax.text(x, y - 2, str(display_id), fontsize=5, color='white', 
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.4),
                       ha='left', va='bottom', weight='bold')
    
    # Apply the single overlay
    ax.imshow(overlay)
    
    # Enhanced title with more information
    title_text = f'{title}\n({len(masks)} masks'
    if show_numbers:
        title_text += f', showing numbers for first {min(max_numbers, len(masks))} masks'
    title_text += ')'
    
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')  # Higher DPI for better text readability
    plt.close()
    
    print(f"Saved visualization: {output_path}")

def save_comprehensive_metadata(all_masks, filtered_masks, clustered_masks, cluster_labels, output_dir, removed_edge=None, removed_circularity=None, removed_multi_blob=None):
    """Save comprehensive metadata about all masks and clustering results"""
    
    # Calculate statistics for all masks
    all_areas = [mask['area'] for mask in all_masks]
    all_circularities = [mask.get('circularity', 0.0) for mask in all_masks]
    all_stabilities = [mask['stability_score'] for mask in all_masks]
    
    metadata = {
        'processing_summary': {
            'original_masks': len(all_masks),
            'after_edge_filtering': len(all_masks) - len(removed_edge) if removed_edge else len(all_masks),
            'after_circularity_filtering': (len(all_masks) - len(removed_edge) if removed_edge else len(all_masks)) - len(removed_circularity) if removed_circularity else len(all_masks) - len(removed_edge) if removed_edge else len(all_masks),
            'after_blob_filtering': len(filtered_masks),
            'removed_edge_masks': len(removed_edge) if removed_edge else 0,
            'removed_low_circularity_masks': len(removed_circularity) if removed_circularity else 0,
            'removed_distant_blob_masks': len(removed_multi_blob) if removed_multi_blob else 0,
            'final_clustered_masks': len(filtered_masks)
        },
        'filtering_criteria': {
            'edge_threshold': 5,
            'min_circularity': 0.53,
            'max_blob_distance': 50
        },
        'clustering_info': {
            'method': 'K-means',
            'n_clusters': len(clustered_masks),
            'features_used': ['area', 'bbox_width', 'bbox_height', 'aspect_ratio', 'stability_score', 'circularity']
        },
        'overall_statistics': {
            'area': {
                'min': float(min(all_areas)),
                'max': float(max(all_areas)),
                'mean': float(np.mean(all_areas)),
                'median': float(np.median(all_areas)),
                'std': float(np.std(all_areas))
            },
            'circularity': {
                'min': float(min(all_circularities)),
                'max': float(max(all_circularities)),
                'mean': float(np.mean(all_circularities)),
                'median': float(np.median(all_circularities)),
                'std': float(np.std(all_circularities))
            },
            'stability_score': {
                'min': float(min(all_stabilities)),
                'max': float(max(all_stabilities)),
                'mean': float(np.mean(all_stabilities)),
                'median': float(np.median(all_stabilities)),
                'std': float(np.std(all_stabilities))
            }
        },
        'all_masks_details': [],
        'clusters': []
    }
    
        # Create sets for faster lookup
    removed_edge_set = set(id(mask) for mask in (removed_edge or []))
    removed_circularity_set = set(id(mask) for mask in (removed_circularity or []))
    removed_distant_blob_set = set(id(mask) for mask in (removed_multi_blob or []))
    filtered_masks_set = set(id(mask) for mask in filtered_masks)
    
            # Add detailed information for each mask
    for i, mask_data in enumerate(all_masks):
        bbox = mask_data['bbox']
        mask_detail = {
            'id': mask_data['original_sam_id'],  # Use original SAM ID
            'area': mask_data['area'],
            'bbox': bbox,  # [x, y, width, height]
            'bbox_center': [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2],
            'aspect_ratio': bbox[2] / bbox[3] if bbox[3] > 0 else 1.0,
            'stability_score': mask_data['stability_score'],
            'circularity': mask_data.get('circularity', 0.0),
            'blob_distance': mask_data.get('blob_distance', 0.0),
            'predicted_iou': mask_data.get('predicted_iou', 0.0),
            'filtering_status': {
                'edge_touching': id(mask_data) in removed_edge_set,
                'low_circularity': id(mask_data) in removed_circularity_set,
                'distant_blob': id(mask_data) in removed_distant_blob_set,
                'included_in_clustering': id(mask_data) in filtered_masks_set
            }
        }
        metadata['all_masks_details'].append(mask_detail)
    
    # Add cluster information
    for i, cluster in enumerate(clustered_masks):
        areas = [mask['area'] for mask in cluster]
        circularities = [mask.get('circularity', 0.0) for mask in cluster]
        stabilities = [mask['stability_score'] for mask in cluster]
        widths = [mask['bbox'][2] for mask in cluster]
        heights = [mask['bbox'][3] for mask in cluster]
        
        cluster_info = {
            'cluster_id': i,
            'mask_count': len(cluster),
            'mask_ids': [filtered_masks[j]['original_sam_id'] for j, label in enumerate(cluster_labels) if label == i],
            'statistics': {
                'area': {
                    'min': float(min(areas)) if areas else 0,
                    'max': float(max(areas)) if areas else 0,
                    'mean': float(np.mean(areas)) if areas else 0,
                    'std': float(np.std(areas)) if areas else 0
                },
                'circularity': {
                    'min': float(min(circularities)) if circularities else 0,
                    'max': float(max(circularities)) if circularities else 0,
                    'mean': float(np.mean(circularities)) if circularities else 0,
                    'std': float(np.std(circularities)) if circularities else 0
                },
                'stability_score': {
                    'min': float(min(stabilities)) if stabilities else 0,
                    'max': float(max(stabilities)) if stabilities else 0,
                    'mean': float(np.mean(stabilities)) if stabilities else 0,
                    'std': float(np.std(stabilities)) if stabilities else 0
                },
                'size': {
                    'width_mean': float(np.mean(widths)) if widths else 0,
                    'height_mean': float(np.mean(heights)) if heights else 0,
                    'width_std': float(np.std(widths)) if widths else 0,
                    'height_std': float(np.std(heights)) if heights else 0
                }
            }
        }
        metadata['clusters'].append(cluster_info)
    
    # Save comprehensive metadata
    metadata_path = os.path.join(output_dir, 'comprehensive_masks_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved comprehensive metadata: {metadata_path}")
    
    # Also save a summary CSV for easy analysis
    import csv
    csv_path = os.path.join(output_dir, 'masks_summary.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['mask_id', 'area', 'circularity', 'blob_distance', 'stability_score', 'bbox_x', 'bbox_y', 
                     'bbox_width', 'bbox_height', 'aspect_ratio', 'cluster_id', 'edge_touching', 
                     'low_circularity', 'distant_blob', 'included_in_clustering']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for mask_detail in metadata['all_masks_details']:
            cluster_id = -1
            for i, cluster_info in enumerate(metadata['clusters']):
                if mask_detail['id'] in cluster_info['mask_ids']:
                    cluster_id = i
                    break
            
            writer.writerow({
                'mask_id': mask_detail['id'],
                'area': mask_detail['area'],
                'circularity': mask_detail['circularity'],
                'blob_distance': mask_detail['blob_distance'],
                'stability_score': mask_detail['stability_score'],
                'bbox_x': mask_detail['bbox'][0],
                'bbox_y': mask_detail['bbox'][1],
                'bbox_width': mask_detail['bbox'][2],
                'bbox_height': mask_detail['bbox'][3],
                'aspect_ratio': mask_detail['aspect_ratio'],
                'cluster_id': cluster_id,
                'edge_touching': mask_detail['filtering_status']['edge_touching'],
                'low_circularity': mask_detail['filtering_status']['low_circularity'],
                'distant_blob': mask_detail['filtering_status']['distant_blob'],
                'included_in_clustering': mask_detail['filtering_status']['included_in_clustering']
            })
    
    print(f"Saved summary CSV: {csv_path}")

def is_mask_touching_edge(mask, bbox, image_shape, edge_threshold=5):
    """Check if mask is touching the image edges"""
    height, width = image_shape[:2]
    x, y, w, h = bbox
    
    # Check if bounding box is close to edges
    touching_left = x <= edge_threshold
    touching_right = (x + w) >= (width - edge_threshold)
    touching_top = y <= edge_threshold
    touching_bottom = (y + h) >= (height - edge_threshold)
    
    return touching_left or touching_right or touching_top or touching_bottom

def filter_problematic_masks(masks, image_shape, edge_threshold=5):
    """Filter out only edge-touching masks (no longer filtering large masks)"""
    print("Preprocessing masks: filtering out edge-touching masks only...")
    
    original_count = len(masks)
    
    filtered_masks = []
    removed_edge = []
    
    for mask_data in masks:
        mask = mask_data['segmentation']
        bbox = mask_data['bbox']
        
        # Check if mask is touching edges
        is_touching_edge = is_mask_touching_edge(mask, bbox, image_shape, edge_threshold)
        
        # Filter logic - only remove edge-touching masks
        if is_touching_edge:
            removed_edge.append(mask_data)
        else:
            filtered_masks.append(mask_data)
    
    print(f"Filtering results:")
    print(f"  Original masks: {original_count}")
    print(f"  Removed edge-touching masks: {len(removed_edge)}")
    print(f"  Remaining masks: {len(filtered_masks)}")
    
    return filtered_masks, removed_edge

def create_filtered_masks_visualization(image, removed_edge, output_dir):
    """Create visualizations showing the filtered-out edge masks"""
    print("Creating filtered masks visualizations...")
    
    # Edge-touching masks visualization
    if removed_edge:
        create_optimized_visualization(
            image, removed_edge,
            f"Filtered Out: Edge-touching Masks ({len(removed_edge)} masks)",
            os.path.join(output_dir, "filtered_edge_masks.png"),
            colors=plt.cm.Blues(np.linspace(0.3, 1, len(removed_edge))),
            show_numbers=True,
            max_numbers=len(removed_edge),  # Show ALL numbers
            original_indices=[mask['original_sam_id'] for mask in removed_edge]
        )

def save_individual_masks(image, masks, output_folder, category_name, original_indices=None):
    """Save each mask as an individual PNG file in a category folder"""
    import os
    from PIL import Image
    
    # Create category folder
    category_dir = os.path.join(output_folder, f"individual_masks_{category_name}")
    os.makedirs(category_dir, exist_ok=True)
    
    print(f"Saving {len(masks)} individual masks to {category_dir}...")
    
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        bbox = mask_data['bbox']
        
        # Use original SAM ID if available, otherwise use index
        mask_id = original_indices[i] if original_indices is not None else i
        
        # Create a colored mask visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Original image crop with bounding box
        x, y, w, h = bbox
        # Add padding around the bounding box
        padding = 20
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        cropped_image = image[y_start:y_end, x_start:x_end]
        ax1.imshow(cropped_image)
        ax1.set_title(f'Original Image\nMask ID: {mask_id}')
        ax1.axis('off')
        
        # Draw bounding box on cropped image
        rect_x = x - x_start
        rect_y = y - y_start
        rect = patches.Rectangle((rect_x, rect_y), w, h, linewidth=2, 
                               edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        
        # 2. Mask overlay on cropped image
        cropped_mask = mask[y_start:y_end, x_start:x_end]
        ax2.imshow(cropped_image)
        # Create colored overlay
        overlay = np.zeros((*cropped_mask.shape, 4))
        overlay[cropped_mask] = [0, 1, 0, 0.6]  # Green with transparency
        ax2.imshow(overlay)
        ax2.set_title(f'Mask Overlay\nArea: {mask_data["area"]}')
        ax2.axis('off')
        
        # 3. Pure mask (black background, white mask)
        pure_mask = np.zeros_like(cropped_image)
        pure_mask[cropped_mask] = [255, 255, 255]  # White mask on black background
        ax3.imshow(pure_mask)
        ax3.set_title(f'Pure Mask\nCircularity: {mask_data.get("circularity", 0.0):.3f}')
        ax3.axis('off')
        
        # Add metadata text
        metadata_text = f"""Mask ID: {mask_id}
Area: {mask_data['area']}
Bbox: [{x}, {y}, {w}, {h}]
Circularity: {mask_data.get('circularity', 0.0):.3f}
Stability: {mask_data.get('stability_score', 0.0):.3f}
Category: {category_name}"""
        
        plt.figtext(0.02, 0.02, metadata_text, fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the individual mask
        mask_filename = f"mask_{mask_id:04d}_{category_name}.png"
        mask_path = os.path.join(category_dir, mask_filename)
        plt.savefig(mask_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if i % 50 == 0:
            print(f"Saved {i+1}/{len(masks)} masks...")
    
    print(f"✅ Saved all {len(masks)} individual masks to {category_dir}")

def save_all_individual_masks(image, all_masks, filtered_masks, removed_edge, removed_circularity, removed_multi_blob, clustered_masks, output_dir):
    """Save individual masks organized by categories"""
    print("\n" + "="*50)
    print("SAVING INDIVIDUAL MASKS BY CATEGORY")
    print("="*50)
    
    # Create base directory for individual masks
    individual_masks_base = os.path.join(output_dir, "individual_masks")
    os.makedirs(individual_masks_base, exist_ok=True)
    
    # 1. All masks that made it to clustering (kmeans)
    if filtered_masks:
        save_individual_masks(
            image, filtered_masks, individual_masks_base, "all_masks_kmeans",
            original_indices=[mask['original_sam_id'] for mask in filtered_masks]
        )
    
    # 2. Cluster 0 masks
    if clustered_masks and len(clustered_masks[0]) > 0:
        save_individual_masks(
            image, clustered_masks[0], individual_masks_base, "cluster0",
            original_indices=[mask['original_sam_id'] for mask in clustered_masks[0]]
        )
    
    # 3. Cluster 1 masks
    if clustered_masks and len(clustered_masks[1]) > 0:
        save_individual_masks(
            image, clustered_masks[1], individual_masks_base, "cluster1",
            original_indices=[mask['original_sam_id'] for mask in clustered_masks[1]]
        )
    
    # 4. Filtered out - Edge touching
    if removed_edge:
        save_individual_masks(
            image, removed_edge, individual_masks_base, "filtered_edge",
            original_indices=[mask['original_sam_id'] for mask in removed_edge]
        )
    
    # 5. Filtered out - Low circularity
    if removed_circularity:
        save_individual_masks(
            image, removed_circularity, individual_masks_base, "filtered_low_circularity",
            original_indices=[mask['original_sam_id'] for mask in removed_circularity]
        )
    
    # 6. Filtered out - Multi-blob
    if removed_multi_blob:
        save_individual_masks(
            image, removed_multi_blob, individual_masks_base, "filtered_multi_blob",
            original_indices=[mask['original_sam_id'] for mask in removed_multi_blob]
        )
    
    print("\n" + "="*50)
    print("INDIVIDUAL MASK FOLDERS CREATED:")
    print("="*50)
    folders = [
        f"individual_masks_all_masks_kmeans ({len(filtered_masks)} masks)",
        f"individual_masks_cluster0 ({len(clustered_masks[0])} masks)" if clustered_masks else "individual_masks_cluster0 (0 masks)",
        f"individual_masks_cluster1 ({len(clustered_masks[1])} masks)" if clustered_masks else "individual_masks_cluster1 (0 masks)",
        f"individual_masks_filtered_edge ({len(removed_edge)} masks)" if removed_edge else "individual_masks_filtered_edge (0 masks)",
        f"individual_masks_filtered_low_circularity ({len(removed_circularity)} masks)" if removed_circularity else "individual_masks_filtered_low_circularity (0 masks)",
        f"individual_masks_filtered_multi_blob ({len(removed_multi_blob)} masks)" if removed_multi_blob else "individual_masks_filtered_multi_blob (0 masks)"
    ]
    
    for folder in folders:
        print(f"📁 {folder}")
    print("="*50)

def main():
    # Configuration
    model_type = "vit_h"  # Using the base model (smaller, less GPU memory)
    crop_n_layers = 3
    
    # Paths
    image_path = "image/Test/GY_image.png"
    base_output_dir = "image/Output"
    
    # Create output directory with model and layer info
    output_folder_name = f"{model_type}_layer_{crop_n_layers}_180625_circularity_0.53_blob_80_masks"
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
    
    # Generate masks using SAM with model_h
    all_masks = generate_masks(image, model_type=model_type, crop_n_layers=crop_n_layers)
    
    # Add circularity information to all masks and preserve original indices
    all_masks = add_circularity_to_masks(all_masks)
    
    # Add original SAM indices to each mask for tracking through the pipeline
    for i, mask in enumerate(all_masks):
        mask['original_sam_id'] = i
    
    # Step 1: Filter out edge-touching masks
    edge_filtered_masks, removed_edge = filter_problematic_masks(all_masks, image.shape)
    
    # Step 2: Filter by circularity (remove masks with low circularity)
    circularity_filtered_masks, removed_circularity = filter_by_circularity(edge_filtered_masks, min_circularity=0.53)
    
    # Step 3: Filter by blob distance (remove masks with widely separated blobs)
    filtered_masks, removed_multi_blob = filter_by_blob_distance(circularity_filtered_masks, max_distance=80)
    
    # Create visualizations for filtered-out masks
    create_filtered_masks_visualization(image, removed_edge, output_dir)
    if removed_circularity:
        create_optimized_visualization(
            image, removed_circularity,
            f"Filtered Out: Low Circularity Masks ({len(removed_circularity)} masks)",
            os.path.join(output_dir, "filtered_low_circularity_masks.png"),
            colors=plt.cm.Reds(np.linspace(0.3, 1, len(removed_circularity))),
            show_numbers=True,
            max_numbers=len(removed_circularity),  # Show ALL numbers
            original_indices=[mask['original_sam_id'] for mask in removed_circularity]
        )
    if removed_multi_blob:
        create_optimized_visualization(
            image, removed_multi_blob,
            f"Filtered Out: Distant-Blob Masks ({len(removed_multi_blob)} masks)",
            os.path.join(output_dir, "filtered_distant_blob_masks.png"),
            colors=plt.cm.Oranges(np.linspace(0.3, 1, len(removed_multi_blob))),
            show_numbers=True,
            max_numbers=len(removed_multi_blob),  # Show ALL numbers
            original_indices=[mask['original_sam_id'] for mask in removed_multi_blob]
        )
    
    # Cluster filtered masks using K-means
    clustered_masks, cluster_labels = cluster_masks_kmeans(filtered_masks, n_clusters=2)
    
    # Create color schemes for clusters
    all_colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_masks)))
    cluster_colors = [
        plt.cm.Blues(np.linspace(0.3, 1, len(clustered_masks[0]))),
        plt.cm.Reds(np.linspace(0.3, 1, len(clustered_masks[1])))
    ]
    
    # Create optimized visualizations for clustered masks with mask numbers
    print("\nCreating optimized visualizations for clustered masks...")
    
    # 1. All filtered masks with bounding boxes and ALL numbers
    create_optimized_visualization(
        image, filtered_masks, 
        f"All Masks with Numbers (K-means Clustering)\nModel: {model_type}, Layers: {crop_n_layers}", 
        os.path.join(output_dir, "all_masks_kmeans_bboxes.png"),
        colors=all_colors,
        show_numbers=True,
        max_numbers=len(filtered_masks),  # Show ALL mask numbers
        original_indices=[mask['original_sam_id'] for mask in filtered_masks]
    )
    
    # 2. Cluster 0 masks with bounding boxes and ALL numbers
    create_optimized_visualization(
        image, clustered_masks[0], 
        f"Cluster 0 Masks with Numbers\nModel: {model_type}, Layers: {crop_n_layers}", 
        os.path.join(output_dir, "cluster0_masks_bboxes.png"),
        colors=cluster_colors[0],
        show_numbers=True,
        max_numbers=len(clustered_masks[0]),  # Show ALL cluster 0 mask numbers
        original_indices=[mask['original_sam_id'] for mask in clustered_masks[0]]
    )
    
    # 3. Cluster 1 masks with bounding boxes and ALL numbers
    create_optimized_visualization(
        image, clustered_masks[1], 
        f"Cluster 1 Masks with Numbers\nModel: {model_type}, Layers: {crop_n_layers}", 
        os.path.join(output_dir, "cluster1_masks_bboxes.png"),
        colors=cluster_colors[1],
        show_numbers=True,
        max_numbers=len(clustered_masks[1]),  # Show ALL cluster 1 mask numbers
        original_indices=[mask['original_sam_id'] for mask in clustered_masks[1]]
    )
    
    # Save comprehensive metadata
    save_comprehensive_metadata(all_masks, filtered_masks, clustered_masks, cluster_labels, output_dir, removed_edge, removed_circularity, removed_multi_blob)
    
    # Save individual masks
    save_all_individual_masks(image, all_masks, filtered_masks, removed_edge, removed_circularity, removed_multi_blob, clustered_masks, output_dir)
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print(f"Model used: {model_type}")
    print(f"Crop layers: {crop_n_layers}")
    print(f"Output directory: {output_dir}")
    print(f"Original masks generated: {len(all_masks)}")
    print(f"After edge filtering: {len(edge_filtered_masks)}")
    print(f"After circularity filtering: {len(circularity_filtered_masks)}")
    print(f"After blob distance filtering: {len(filtered_masks)}")
    print(f"Filtered out - Edge-touching masks: {len(removed_edge)}")
    print(f"Filtered out - Low circularity masks: {len(removed_circularity)}")
    print(f"Filtered out - Distant-blob masks: {len(removed_multi_blob)}")
    print(f"Clustering method: K-means (2 clusters)")
    print(f"Cluster 0: {len(clustered_masks[0])} masks")
    print(f"Cluster 1: {len(clustered_masks[1])} masks")
    
    # Show circularity statistics
    all_circularities = [mask.get('circularity', 0.0) for mask in all_masks]
    filtered_circularities = [mask.get('circularity', 0.0) for mask in filtered_masks]
    print(f"\nCircularity statistics:")
    print(f"  All masks - mean: {np.mean(all_circularities):.3f}, min: {min(all_circularities):.3f}, max: {max(all_circularities):.3f}")
    print(f"  Filtered masks - mean: {np.mean(filtered_circularities):.3f}, min: {min(filtered_circularities):.3f}, max: {max(filtered_circularities):.3f}")
    
    print("\nPreprocessing applied:")
    print("- Calculated circularity for all masks")
    print("- Filtered out edge-touching masks (incomplete masks)")
    print("- Filtered out low circularity masks (min_circularity: 0.53)")
    print("- Filtered out distant-blob masks (max_distance: 50px)")
    print("\nOptimizations applied:")
    print(f"- Used {model_type} model for better segmentation")
    print(f"- Used crop_n_layers={crop_n_layers} for detailed segmentation")
    print("- K-means clustering with 6 features (including circularity)")
    print("- Enhanced visualizations with mask numbers and info")
    print("- Comprehensive metadata generation")
    print("\nOutput files created:")
    print("CLUSTERED MASKS WITH NUMBERS:")
    print("- all_masks_kmeans_bboxes.png (shows mask IDs, areas, circularity)")
    print("- cluster0_masks_bboxes.png (first 50 masks with details)")
    print("- cluster1_masks_bboxes.png (first 50 masks with details)")
    print("FILTERED OUT MASKS:")
    if removed_edge:
        print("- filtered_edge_masks.png")
    if removed_circularity:
        print("- filtered_low_circularity_masks.png")
    if removed_multi_blob:
        print("- filtered_distant_blob_masks.png")
    print("METADATA & ANALYSIS:")
    print("- comprehensive_masks_metadata.json (detailed JSON metadata)")
    print("- masks_summary.csv (CSV for easy analysis)")
    print("INDIVIDUAL MASKS:")
    print("- all_masks_kmeans (all masks with kmeans clustering)")
    print("- cluster0 (cluster 0 masks)")
    print("- cluster1 (cluster 1 masks)")
    print("- filtered_edge (edge-touching masks)")
    print("- filtered_low_circularity (low circularity masks)")
    print("- filtered_multi_blob (multi-blob masks)")
    print("="*70)

if __name__ == "__main__":
    main() 