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

def save_comprehensive_metadata(all_masks, filtered_masks, clustered_masks, cluster_labels, output_dir, removed_edge=None, removed_circularity=None, removed_multi_blob=None, removed_duplicates_per_cluster=None):
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
            'after_duplicate_removal': sum(len(cluster) for cluster in clustered_masks),
            'removed_edge_masks': len(removed_edge) if removed_edge else 0,
            'removed_low_circularity_masks': len(removed_circularity) if removed_circularity else 0,
            'removed_distant_blob_masks': len(removed_multi_blob) if removed_multi_blob else 0,
            'removed_duplicate_masks': sum(len(removed) for removed in removed_duplicates_per_cluster) if removed_duplicates_per_cluster else 0,
            'final_clustered_masks': sum(len(cluster) for cluster in clustered_masks)
        },
        'filtering_criteria': {
            'edge_threshold': 5,
            'min_circularity': 0.53,
            'max_blob_distance': 50,
            'duplicate_removal_overlap_threshold': 60.0,
            'duplicate_removal_strategy': 'remove_larger_mask'
        },
        'clustering_info': {
            'method': 'K-means',
            'n_clusters': len(clustered_masks),
            'features_used': ['area', 'bbox_width', 'bbox_height', 'aspect_ratio', 'stability_score', 'circularity'],
            'post_clustering_duplicate_removal': True,
            'duplicate_removal_details': [len(removed) for removed in removed_duplicates_per_cluster] if removed_duplicates_per_cluster else []
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
    
    # Create set for duplicate removal tracking
    removed_duplicates_set = set()
    if removed_duplicates_per_cluster:
        for removed_cluster in removed_duplicates_per_cluster:
            for mask in removed_cluster:
                removed_duplicates_set.add(id(mask))
    
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
                'included_in_clustering': id(mask_data) in filtered_masks_set,
                'removed_as_duplicate': id(mask_data) in removed_duplicates_set
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
                     'low_circularity', 'distant_blob', 'included_in_clustering', 'removed_as_duplicate']
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
                'included_in_clustering': mask_detail['filtering_status']['included_in_clustering'],
                'removed_as_duplicate': mask_detail['filtering_status']['removed_as_duplicate']
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

def calculate_bbox_overlap_percentage(bbox1, bbox2):
    """Calculate the percentage of bbox1 that overlaps with bbox2 using bounding box coordinates."""
    # bbox format: [x, y, width, height]
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate bbox1 area
    bbox1_area = w1 * h1
    if bbox1_area == 0:
        return 0.0
    
    # Calculate intersection coordinates
    # Left edge of intersection
    left = max(x1, x2)
    # Right edge of intersection  
    right = min(x1 + w1, x2 + w2)
    # Top edge of intersection
    top = max(y1, y2)
    # Bottom edge of intersection
    bottom = min(y1 + h1, y2 + h2)
    
    # Check if there's actually an intersection
    if left >= right or top >= bottom:
        return 0.0
    
    # Calculate intersection area
    intersection_width = right - left
    intersection_height = bottom - top
    intersection_area = intersection_width * intersection_height
    
    # Calculate percentage of bbox1 covered by bbox2
    overlap_percentage = (intersection_area / bbox1_area) * 100.0
    return overlap_percentage

def analyze_mask_overlaps(clustered_masks, image, output_dir, min_overlap_percentage=80.0):
    """
    Analyze overlaps between smaller masks and larger masks with second-level clustering.
    Returns larger masks labeled with the number of overlapping smaller masks or "aggregated overlap".
    """
    print("\n" + "="*60)
    print("OVERLAP ANALYSIS: Finding smaller masks that overlap with larger masks")
    print("="*60)
    
    if len(clustered_masks) < 2:
        print("Not enough clusters for overlap analysis")
        return {'final_masks': [], 'overlap_summary': {}}
    
    cluster_0 = clustered_masks[0]
    cluster_1 = clustered_masks[1]
    
    # Determine which cluster has larger masks on average
    if cluster_0 and cluster_1:
        avg_area_0 = np.mean([mask['area'] for mask in cluster_0])
        avg_area_1 = np.mean([mask['area'] for mask in cluster_1])
        
        if avg_area_0 > avg_area_1:
            larger_masks = cluster_0
            smaller_masks = cluster_1
            print(f"Cluster 0 identified as LARGER masks (avg area: {avg_area_0:.0f})")
            print(f"Cluster 1 identified as SMALLER masks (avg area: {avg_area_1:.0f})")
        else:
            larger_masks = cluster_1
            smaller_masks = cluster_0
            print(f"Cluster 1 identified as LARGER masks (avg area: {avg_area_1:.0f})")
            print(f"Cluster 0 identified as SMALLER masks (avg area: {avg_area_0:.0f})")
    else:
        print("One cluster is empty, using available cluster as larger masks")
        larger_masks = cluster_0 if cluster_0 else cluster_1
        smaller_masks = cluster_1 if cluster_0 else cluster_0
    
    # NEW: Perform second-level K-means clustering on smaller masks
    clustered_smaller_masks, smaller_cluster_labels = cluster_smaller_masks_kmeans(smaller_masks, n_clusters=2)
    
    # Determine which second-level cluster contains smaller vs larger masks
    if len(clustered_smaller_masks) >= 2:
        avg_area_sc0 = np.mean([mask['area'] for mask in clustered_smaller_masks[0]]) if clustered_smaller_masks[0] else 0
        avg_area_sc1 = np.mean([mask['area'] for mask in clustered_smaller_masks[1]]) if clustered_smaller_masks[1] else 0
        
        if avg_area_sc0 < avg_area_sc1:
            smaller_group_id = 0  # SC0 contains smaller masks
            larger_group_id = 1   # SC1 contains larger masks
            print(f"SC0 identified as SMALLER group (avg area: {avg_area_sc0:.0f})")
            print(f"SC1 identified as LARGER group (avg area: {avg_area_sc1:.0f})")
        else:
            smaller_group_id = 1  # SC1 contains smaller masks
            larger_group_id = 0   # SC0 contains larger masks
            print(f"SC1 identified as SMALLER group (avg area: {avg_area_sc1:.0f})")
            print(f"SC0 identified as LARGER group (avg area: {avg_area_sc0:.0f})")
    else:
        smaller_group_id = 0
        larger_group_id = 1
        print("Using default assignment: SC0=smaller, SC1=larger")
    
    print(f"Analyzing overlaps with minimum {min_overlap_percentage}% coverage...")
    print(f"Large masks to analyze: {len(larger_masks)}")
    print(f"Small masks to check for overlap: {len(smaller_masks)}")
    print(f"Second-level small clusters: {[len(c) for c in clustered_smaller_masks]}")
    
    # Analyze each larger mask for overlaps with smaller masks
    final_masks = []
    overlap_details = []
    
    for i, large_mask in enumerate(larger_masks):
        large_bbox = large_mask['bbox']
        overlapping_small_masks = []
        overlap_percentages = []
        overlapping_smaller_group = []  # Masks from the smaller second-level group
        overlapping_larger_group = []   # Masks from the larger second-level group
        
        # Check overlap with each smaller mask
        for j, small_mask in enumerate(smaller_masks):
            small_bbox = small_mask['bbox']
            
            # Calculate overlap percentage using bounding boxes (what % of small bbox is covered by large bbox)
            overlap_pct = calculate_bbox_overlap_percentage(small_bbox, large_bbox)
            
            if overlap_pct >= min_overlap_percentage:
                # Find which small cluster this mask belongs to
                small_cluster_id = smaller_cluster_labels[j] if j < len(smaller_cluster_labels) else 0
                
                overlap_info = {
                    'small_mask_id': small_mask['original_sam_id'],
                    'small_mask_index': j,
                    'small_cluster_id': small_cluster_id,
                    'overlap_percentage': overlap_pct,
                    'small_mask_area': small_mask['area']
                }
                
                overlapping_small_masks.append(overlap_info)
                overlap_percentages.append(overlap_pct)
                
                # Separate into smaller vs larger groups based on second-level clustering
                if small_cluster_id == smaller_group_id:
                    overlapping_smaller_group.append(overlap_info)
                elif small_cluster_id == larger_group_id:
                    overlapping_larger_group.append(overlap_info)
        
        # Refined overlap classification logic
        num_smaller_overlaps = len(overlapping_smaller_group)
        num_larger_overlaps = len(overlapping_larger_group)
        total_overlaps = len(overlapping_small_masks)
        
        # Determine overlap type based on refined logic
        if num_larger_overlaps > 0:
            # Contains larger masks from second clustering = Aggregate group
            is_aggregate = True
            overlap_type = 'aggregate'
            overlap_label = f"AGG({total_overlaps})"
        else:
            # Contains only smaller masks from second clustering = Simple count
            is_aggregate = False
            overlap_type = 'simple'
            overlap_label = f"{total_overlaps}"
        
        final_mask = large_mask.copy()
        final_mask['overlap_count'] = total_overlaps
        final_mask['overlapping_masks'] = overlapping_small_masks
        final_mask['overlapping_smaller_group'] = overlapping_smaller_group
        final_mask['overlapping_larger_group'] = overlapping_larger_group
        final_mask['num_smaller_overlaps'] = num_smaller_overlaps
        final_mask['num_larger_overlaps'] = num_larger_overlaps
        final_mask['is_aggregate_overlap'] = is_aggregate
        final_mask['overlap_type'] = overlap_type
        final_mask['overlap_label'] = overlap_label
        final_mask['smaller_group_id'] = smaller_group_id
        final_mask['larger_group_id'] = larger_group_id
        
        final_masks.append(final_mask)
        
        # Store detailed information
        overlap_detail = {
            'large_mask_id': large_mask['original_sam_id'],
            'large_mask_area': large_mask['area'],
            'overlap_count': total_overlaps,
            'num_smaller_overlaps': num_smaller_overlaps,
            'num_larger_overlaps': num_larger_overlaps,
            'overlapping_small_masks': overlapping_small_masks,
            'overlapping_smaller_group': overlapping_smaller_group,
            'overlapping_larger_group': overlapping_larger_group,
            'is_aggregate_overlap': is_aggregate,
            'overlap_type': overlap_type,
            'smaller_group_id': smaller_group_id,
            'larger_group_id': larger_group_id,
            'avg_overlap_percentage': np.mean(overlap_percentages) if overlap_percentages else 0.0
        }
        overlap_details.append(overlap_detail)
        
        if i < 10:  # Print details for first 10 masks
            overlap_type_str = "AGGREGATE" if is_aggregate else "SIMPLE"
            print(f"Large mask {large_mask['original_sam_id']}: {total_overlaps} overlapping masks ({overlap_type_str})")
            print(f"  - Smaller group overlaps: {num_smaller_overlaps}")
            print(f"  - Larger group overlaps: {num_larger_overlaps}")
            if overlapping_small_masks:
                for overlap in overlapping_small_masks:
                    group_type = "SMALLER" if overlap['small_cluster_id'] == smaller_group_id else "LARGER"
                    print(f"  - Small mask {overlap['small_mask_id']} (SC{overlap['small_cluster_id']}-{group_type}): {overlap['overlap_percentage']:.1f}% overlap")
    
    # Create summary statistics
    overlap_counts = [mask['overlap_count'] for mask in final_masks]
    unique_counts, count_frequencies = np.unique(overlap_counts, return_counts=True)
    
    # Separate aggregate and simple overlaps based on refined logic
    aggregate_masks = [mask for mask in final_masks if mask['is_aggregate_overlap']]
    simple_masks = [mask for mask in final_masks if not mask['is_aggregate_overlap']]
    
    print(f"\nREFINED OVERLAP SUMMARY:")
    print(f"Total large masks analyzed: {len(final_masks)}")
    print(f"  Simple overlaps (only smaller group): {len(simple_masks)} masks")
    print(f"  Aggregate overlaps (contains larger group): {len(aggregate_masks)} masks")
    for count, freq in zip(unique_counts, count_frequencies):
        print(f"  {freq} large masks have {count} total overlapping masks")
    
    # Show breakdown by smaller vs larger group overlaps
    if final_masks:
        smaller_group_overlaps = [mask['num_smaller_overlaps'] for mask in final_masks]
        larger_group_overlaps = [mask['num_larger_overlaps'] for mask in final_masks]
        print(f"\nDETAILED BREAKDOWN:")
        print(f"  Total smaller group overlaps: {sum(smaller_group_overlaps)}")
        print(f"  Total larger group overlaps: {sum(larger_group_overlaps)}")
    
    if aggregate_masks:
        print(f"\nAGGREGATE GROUP DETAILS:")
        for mask in aggregate_masks[:5]:  # Show first 5 aggregate overlaps
            print(f"  Mask {mask['original_sam_id']}: {mask['num_smaller_overlaps']} smaller + {mask['num_larger_overlaps']} larger = {mask['overlap_count']} total")
    
    # Validate final masks before visualization
    final_masks = validate_and_fix_bbox_data(final_masks, "final_masks_for_overlap_visualization")
    
    # Create visualization of final masks with overlap labels
    create_overlap_visualization(final_masks, image, output_dir, min_overlap_percentage)
    
    # Save overlap analysis to CSV
    save_overlap_analysis_csv(overlap_details, output_dir)
    
    overlap_summary = {
        'total_large_masks': len(final_masks),
        'simple_overlaps': len(simple_masks),
        'aggregate_overlaps': len(aggregate_masks),
        'overlap_distribution': dict(zip(unique_counts.tolist(), count_frequencies.tolist())),
        'min_overlap_percentage': min_overlap_percentage,
        'overlap_details': overlap_details,
        'second_level_clustering': {
            'smaller_clusters': [len(c) for c in clustered_smaller_masks],
            'total_smaller_masks': len(smaller_masks),
            'smaller_group_id': smaller_group_id,
            'larger_group_id': larger_group_id,
            'smaller_group_avg_area': avg_area_sc0 if smaller_group_id == 0 else avg_area_sc1,
            'larger_group_avg_area': avg_area_sc1 if smaller_group_id == 0 else avg_area_sc0
        },
        'refined_logic': {
            'total_smaller_group_overlaps': sum([mask['num_smaller_overlaps'] for mask in final_masks]),
            'total_larger_group_overlaps': sum([mask['num_larger_overlaps'] for mask in final_masks])
        }
    }
    
    return {
        'final_masks': final_masks,
        'overlap_summary': overlap_summary,
        'larger_cluster_masks': larger_masks,
        'smaller_cluster_masks': smaller_masks,
        'clustered_smaller_masks': clustered_smaller_masks,
        'smaller_cluster_labels': smaller_cluster_labels,
        'smaller_group_id': smaller_group_id,
        'larger_group_id': larger_group_id
    }

def create_overlap_visualization(final_masks, image, output_dir, min_overlap_percentage):
    """Create visualization showing larger masks labeled with overlap counts."""
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    
    print(f"\nCreating overlap visualization...")
    
    # DEBUG: Check bbox values before visualization
    print(f"DEBUG: Checking bbox values for {len(final_masks)} final masks:")
    if final_masks:
        for i, mask_data in enumerate(final_masks[:3]):  # Check first 3 masks
            bbox = mask_data['bbox']
            print(f"  Mask {i}: bbox = {bbox}, type = {type(bbox)}, values = {[type(x) for x in bbox]}")
            
        # Check for obviously wrong bbox values
        bbox_issues = []
        for i, mask_data in enumerate(final_masks):
            bbox = mask_data['bbox']
            x, y, w, h = bbox
            
            # Convert to int if they're not already (common issue)
            x, y, w, h = int(x), int(y), int(w), int(h)
            mask_data['bbox'] = [x, y, w, h]  # Update with int values
            
            # Check for invalid coordinates
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                bbox_issues.append(f"Mask {i}: Invalid bbox {bbox}")
            elif x > image.shape[1] or y > image.shape[0]:
                bbox_issues.append(f"Mask {i}: Bbox {bbox} outside image bounds {image.shape}")
        
        if bbox_issues:
            print(f"WARNING: Found {len(bbox_issues)} bbox issues:")
            for issue in bbox_issues[:5]:  # Show first 5 issues
                print(f"  {issue}")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(image)
    
    # Create color map for different overlap counts
    max_overlap = max([mask['overlap_count'] for mask in final_masks]) if final_masks else 0
    colors = plt.cm.viridis(np.linspace(0, 1, max_overlap + 1))
    
    # Draw each final mask with its overlap count label
    drawn_masks = 0
    for mask_data in final_masks:
        bbox = mask_data['bbox']
        overlap_count = mask_data['overlap_count']
        is_aggregate = mask_data.get('is_aggregate_overlap', False)
        x, y, w, h = bbox
        
        # Additional validation and conversion
        try:
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Skip masks with invalid dimensions
            if w <= 0 or h <= 0:
                print(f"WARNING: Skipping mask with invalid dimensions: bbox={bbox}")
                continue
                
            # Skip masks outside image bounds
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                print(f"WARNING: Skipping mask outside image bounds: bbox={bbox}, image_shape={image.shape}")
                continue
                
        except (ValueError, TypeError) as e:
            print(f"ERROR: Cannot convert bbox to integers: {bbox}, error: {e}")
            continue
        
        # Color based on overlap count, with different style for aggregate
        if is_aggregate:
            # Use red colors for aggregate overlaps
            color = plt.cm.Reds(0.5 + 0.5 * (overlap_count / max(max_overlap, 1)))
            linewidth = 3  # Thicker border for aggregate
        else:
            # Use original viridis colors for simple overlaps
            color = colors[overlap_count] if overlap_count <= max_overlap else colors[-1]
            linewidth = 2
        
        # Draw bounding box
        rect = Rectangle((x, y), w, h, linewidth=linewidth, 
                        edgecolor=color[:3], facecolor='none')
        ax.add_patch(rect)
        
        # Add overlap count label with aggregation indicator
        overlap_label = mask_data.get('overlap_label', str(overlap_count))
        label_text = f"ID:{mask_data['original_sam_id']}\n{overlap_label}"
        
        # Different background color for aggregate overlaps
        bg_color = 'red' if is_aggregate else 'black'
        ax.text(x, y - 5, label_text, fontsize=6, color='white', 
               bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_color, alpha=0.8),
               ha='left', va='bottom', weight='bold')
        
        drawn_masks += 1
    
    print(f"Successfully drew {drawn_masks}/{len(final_masks)} masks")
    
    # Create title and legend
    aggregate_count = len([m for m in final_masks if m.get('is_aggregate_overlap', False)])
    simple_count = len(final_masks) - aggregate_count
    
    title = f"Large Masks with Refined Overlap Analysis\n(Min {min_overlap_percentage}% overlap, Simple: {simple_count}, Aggregate: {aggregate_count})\nAGG(n) = Aggregate overlap contains larger group masks"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add color bar legend for simple overlaps
    if max_overlap > 0:
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=max_overlap))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('Simple Overlaps Count', rotation=270, labelpad=20)
    
    # Add text legend for aggregate overlaps
    if aggregate_count > 0:
        legend_text = "Legend:\n• Black labels = Simple overlaps\n• Red labels = Aggregate overlaps\n• Thick red borders = Aggregate"
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(output_dir, f"overlap_analysis_masks_{min_overlap_percentage}pct.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved overlap visualization: {output_path}")

def save_overlap_analysis_csv(overlap_details, output_dir):
    """Save detailed overlap analysis to CSV."""
    import csv
    import os
    
    csv_path = os.path.join(output_dir, "overlap_analysis_details.csv")
    
    print(f"Saving overlap analysis to: {csv_path}")
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['large_mask_id', 'large_mask_area', 'overlap_count', 'overlap_type',
                     'is_aggregate', 'num_smaller_overlaps', 'num_larger_overlaps',
                     'smaller_group_id', 'larger_group_id', 'overlapping_small_mask_ids', 
                     'smaller_group_mask_ids', 'larger_group_mask_ids', 'overlap_percentages', 
                     'avg_overlap_percentage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for detail in overlap_details:
            # Format overlapping mask info for CSV
            small_mask_ids = [str(overlap['small_mask_id']) for overlap in detail['overlapping_small_masks']]
            overlap_percentages = [f"{overlap['overlap_percentage']:.1f}" for overlap in detail['overlapping_small_masks']]
            
            # Separate smaller and larger group mask IDs
            smaller_group_ids = [str(overlap['small_mask_id']) for overlap in detail['overlapping_smaller_group']]
            larger_group_ids = [str(overlap['small_mask_id']) for overlap in detail['overlapping_larger_group']]
            
            writer.writerow({
                'large_mask_id': detail['large_mask_id'],
                'large_mask_area': detail['large_mask_area'],
                'overlap_count': detail['overlap_count'],
                'overlap_type': detail['overlap_type'],
                'is_aggregate': detail['is_aggregate_overlap'],
                'num_smaller_overlaps': detail['num_smaller_overlaps'],
                'num_larger_overlaps': detail['num_larger_overlaps'],
                'smaller_group_id': detail['smaller_group_id'],
                'larger_group_id': detail['larger_group_id'],
                'overlapping_small_mask_ids': ';'.join(small_mask_ids),
                'smaller_group_mask_ids': ';'.join(smaller_group_ids),
                'larger_group_mask_ids': ';'.join(larger_group_ids),
                'overlap_percentages': ';'.join(overlap_percentages),
                'avg_overlap_percentage': f"{detail['avg_overlap_percentage']:.1f}"
            })
    
    print(f"Overlap analysis CSV saved with {len(overlap_details)} records")

def save_all_individual_masks(image, all_masks, filtered_masks, removed_edge, removed_circularity, removed_multi_blob, clustered_masks, output_dir, removed_duplicates_per_cluster=None):
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
    
    # 7. Removed duplicates from each cluster
    if removed_duplicates_per_cluster:
        for cluster_idx, removed_duplicates in enumerate(removed_duplicates_per_cluster):
            if removed_duplicates:
                save_individual_masks(
                    image, removed_duplicates, individual_masks_base, f"filtered_duplicates_cluster{cluster_idx}",
                    original_indices=[mask['original_sam_id'] for mask in removed_duplicates]
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
    
    # Add duplicate removal folders to the list
    if removed_duplicates_per_cluster:
        for cluster_idx, removed_duplicates in enumerate(removed_duplicates_per_cluster):
            if removed_duplicates:
                folders.append(f"individual_masks_filtered_duplicates_cluster{cluster_idx} ({len(removed_duplicates)} masks)")
    
    for folder in folders:
        print(f"📁 {folder}")
    print("="*50)

def remove_duplicate_masks_within_clusters(clustered_masks, min_overlap_percentage=20.0):
    """
    Remove duplicate masks within each cluster based on bounding box overlap.
    When two masks overlap more than min_overlap_percentage, the larger mask is removed.
    
    Args:
        clustered_masks: List of clusters, each containing list of mask dictionaries
        min_overlap_percentage: Minimum overlap percentage to consider masks as duplicates
        
    Returns:
        tuple: (cleaned_clustered_masks, removed_duplicates_per_cluster)
    """
    print("\n" + "="*60)
    print(f"REMOVING DUPLICATE MASKS WITHIN CLUSTERS (overlap > {min_overlap_percentage}%)")
    print("="*60)
    
    cleaned_clusters = []
    removed_duplicates_per_cluster = []
    total_removed = 0
    
    for cluster_idx, cluster_masks in enumerate(clustered_masks):
        print(f"\nProcessing Cluster {cluster_idx}: {len(cluster_masks)} masks")
        
        if len(cluster_masks) <= 1:
            # No duplicates possible with 0 or 1 mask
            cleaned_clusters.append(cluster_masks.copy())
            removed_duplicates_per_cluster.append([])
            continue
        
        # Sort masks by area (smallest first) to prioritize keeping smaller masks
        sorted_masks = sorted(cluster_masks, key=lambda x: x['area'], reverse=False)
        
        masks_to_keep = []
        removed_masks = []
        
        for i, current_mask in enumerate(sorted_masks):
            is_duplicate = False
            current_bbox = current_mask['bbox']
            
            # Check overlap with all masks we've already decided to keep
            for kept_mask in masks_to_keep:
                kept_bbox = kept_mask['bbox']
                
                # Calculate overlap percentage (what % of current mask overlaps with kept mask)
                overlap_pct = calculate_bbox_overlap_percentage(current_bbox, kept_bbox)
                
                if overlap_pct >= min_overlap_percentage:
                    # Current mask is a duplicate of a smaller mask we're keeping
                    print(f"  Removing mask {current_mask['original_sam_id']} (area: {current_mask['area']:.0f}) - "
                          f"{overlap_pct:.1f}% overlap with mask {kept_mask['original_sam_id']} (area: {kept_mask['area']:.0f})")
                    removed_masks.append(current_mask)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                masks_to_keep.append(current_mask)
        
        print(f"  Cluster {cluster_idx} results: kept {len(masks_to_keep)}, removed {len(removed_masks)} duplicates")
        
        cleaned_clusters.append(masks_to_keep)
        removed_duplicates_per_cluster.append(removed_masks)
        total_removed += len(removed_masks)
    
    print(f"\nDUPLICATE REMOVAL SUMMARY:")
    print(f"Total masks removed across all clusters: {total_removed}")
    for i, (original_count, cleaned_count, removed_count) in enumerate(
        zip([len(cluster) for cluster in clustered_masks],
            [len(cluster) for cluster in cleaned_clusters], 
            [len(removed) for removed in removed_duplicates_per_cluster])):
        print(f"  Cluster {i}: {original_count} → {cleaned_count} (removed {removed_count})")
    
    return cleaned_clusters, removed_duplicates_per_cluster

def create_removed_duplicates_visualization(image, removed_duplicates_per_cluster, output_dir):
    """Create visualization showing all removed duplicate masks from all clusters."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    print(f"\nCreating removed duplicates visualization...")
    
    # Combine all removed duplicates from all clusters
    all_removed_duplicates = []
    cluster_info = []
    
    for cluster_idx, removed_duplicates in enumerate(removed_duplicates_per_cluster):
        for mask in removed_duplicates:
            all_removed_duplicates.append(mask)
            cluster_info.append(cluster_idx)
    
    if not all_removed_duplicates:
        print("No duplicate masks were removed, skipping visualization.")
        return
    
    print(f"Total removed duplicates to visualize: {len(all_removed_duplicates)}")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(image)
    
    # Create color scheme for different clusters
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Draw each removed duplicate mask
    for i, (mask_data, cluster_idx) in enumerate(zip(all_removed_duplicates, cluster_info)):
        bbox = mask_data['bbox']
        x, y, w, h = bbox
        
        # Color based on cluster
        color = colors[cluster_idx % len(colors)]
        
        # Draw bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color, facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add mask ID label on top of the bounding box (outside)
        label_text = f"ID:{mask_data['original_sam_id']}\nC{cluster_idx}"
        ax.text(x + w/2, y - 5, label_text, fontsize=4, color='white', 
               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.9),
               ha='center', va='bottom', weight='bold')
    
    # Create title and legend
    total_removed = len(all_removed_duplicates)
    cluster_counts = {}
    for cluster_idx in cluster_info:
        cluster_counts[cluster_idx] = cluster_counts.get(cluster_idx, 0) + 1
    
    cluster_summary = ", ".join([f"C{idx}: {count}" for idx, count in cluster_counts.items()])
    title = f"Removed Duplicate Masks - Larger Masks Removed (Total: {total_removed})\n{cluster_summary} masks removed due to >20% overlap"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend for clusters
    legend_elements = []
    for cluster_idx, count in cluster_counts.items():
        color = colors[cluster_idx % len(colors)]
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        label=f'Cluster {cluster_idx} ({count} removed)'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(output_dir, "removed_duplicate_masks.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved removed duplicates visualization: {output_path}")

def create_comprehensive_overlap_visualization(overlap_analysis_results, image, output_dir, min_overlap_percentage):
    """Create visualization showing both small and large masks with second-level clustering and enhanced overlap analysis."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    
    print(f"\nCreating comprehensive overlap visualization with second-level clustering...")
    
    if not overlap_analysis_results or not overlap_analysis_results.get('final_masks'):
        print("No overlap analysis results available, skipping comprehensive visualization.")
        return
    
    final_masks = overlap_analysis_results['final_masks']
    larger_masks = overlap_analysis_results.get('larger_cluster_masks', [])
    smaller_masks = overlap_analysis_results.get('smaller_cluster_masks', [])
    clustered_smaller_masks = overlap_analysis_results.get('clustered_smaller_masks', [])
    smaller_cluster_labels = overlap_analysis_results.get('smaller_cluster_labels', [])
    smaller_group_id = overlap_analysis_results.get('smaller_group_id', 0)
    larger_group_id = overlap_analysis_results.get('larger_group_id', 1)
    
    if not final_masks:
        print("No final masks available for comprehensive visualization.")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.imshow(image)
    
    # Colors for different small clusters with refined logic
    small_cluster_colors = ['cyan', 'magenta']
    group_labels = ['SMALLER', 'LARGER']
    
    # Draw all smaller masks with second-level clustering colors
    print(f"Drawing {len(smaller_masks)} smaller masks with refined clustering...")
    for i, mask_data in enumerate(smaller_masks):
        bbox = mask_data['bbox']
        x, y, w, h = bbox
        
        # Determine which small cluster this mask belongs to
        small_cluster_id = smaller_cluster_labels[i] if i < len(smaller_cluster_labels) else 0
        cluster_color = small_cluster_colors[small_cluster_id % len(small_cluster_colors)]
        
        # Determine if this is from smaller or larger group
        is_smaller_group = (small_cluster_id == smaller_group_id)
        group_label = group_labels[0] if is_smaller_group else group_labels[1]
        
        # Draw smaller mask bounding box with cluster-specific color
        # Use different line styles for smaller vs larger groups
        linestyle = '-' if is_smaller_group else '--'
        rect = patches.Rectangle((x, y), w, h, linewidth=2, linestyle=linestyle,
                               edgecolor=cluster_color, facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add small mask ID label OUTSIDE and ABOVE the bounding box
        label_text = f"SC{small_cluster_id}({group_label}):{mask_data['original_sam_id']}"
        ax.text(x + w/2, y - 3, label_text, fontsize=6, color='white', 
               bbox=dict(boxstyle="round,pad=0.2", facecolor=cluster_color, alpha=0.9),
               ha='center', va='bottom', weight='bold')
    
    # Draw larger masks with refined overlap analysis
    print(f"Drawing {len(final_masks)} larger masks with refined overlap analysis...")
    for mask_data in final_masks:
        bbox = mask_data['bbox']
        overlap_count = mask_data['overlap_count']
        is_aggregate = mask_data.get('is_aggregate_overlap', False)
        overlap_label = mask_data.get('overlap_label', str(overlap_count))
        num_smaller_overlaps = mask_data.get('num_smaller_overlaps', 0)
        num_larger_overlaps = mask_data.get('num_larger_overlaps', 0)
        x, y, w, h = bbox
        
        # Refined color scheme based on overlap type
        if overlap_count == 0:
            color = 'gray'
            alpha = 0.5
            linewidth = 1
        elif is_aggregate:
            # Red colors for aggregate overlaps (contains larger group masks)
            max_overlaps = max([m['overlap_count'] for m in final_masks])
            intensity = min(0.4 + 0.6 * (overlap_count / max(max_overlaps, 1)), 1.0)
            color = (1.0, 1.0 - intensity, 1.0 - intensity)  # Red scale
            alpha = 0.9
            linewidth = 4  # Thick border for aggregate
        else:
            # Green colors for simple overlaps (only smaller group masks)
            max_overlaps = max([m['overlap_count'] for m in final_masks])
            intensity = min(0.3 + 0.7 * (overlap_count / max(max_overlaps, 1)), 1.0)
            color = (1.0 - intensity, 1.0, 1.0 - intensity)  # Green scale
            alpha = 0.8
            linewidth = 2
        
        # Draw larger mask bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=linewidth, 
                               edgecolor=color, facecolor='none', alpha=alpha)
        ax.add_patch(rect)
        
        # Add large mask ID and refined overlap label
        if is_aggregate:
            # Show breakdown for aggregate overlaps
            label_text = f"L:{mask_data['original_sam_id']}\n{overlap_label}\n[S:{num_smaller_overlaps}+L:{num_larger_overlaps}]"
            bg_color = color
        else:
            # Simple overlaps only show total count
            label_text = f"L:{mask_data['original_sam_id']}\n{overlap_label}"
            bg_color = color
        
        ax.text(x, y - 5, label_text, fontsize=7, color='white', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.9),
               ha='left', va='bottom', weight='bold')
    
    # Refined title with second-level clustering information
    total_large = len(final_masks)
    total_small = len(smaller_masks)
    aggregate_count = len([m for m in final_masks if m.get('is_aggregate_overlap', False)])
    simple_count = total_large - aggregate_count
    
    # Get second-level clustering info with refined labeling
    small_cluster_counts = [len(c) for c in clustered_smaller_masks] if clustered_smaller_masks else [0, 0]
    smaller_group_count = small_cluster_counts[smaller_group_id] if smaller_group_id < len(small_cluster_counts) else 0
    larger_group_count = small_cluster_counts[larger_group_id] if larger_group_id < len(small_cluster_counts) else 0
    
    title = (f"Refined Comprehensive Overlap Analysis\n"
             f"Large masks: {total_large} (Simple: {simple_count}, Aggregate: {aggregate_count})\n"
             f"Small masks: {total_small} (SC{smaller_group_id}-SMALLER: {smaller_group_count}, SC{larger_group_id}-LARGER: {larger_group_count})\n"
             f"Min {min_overlap_percentage}% overlap threshold | Refined Logic: Aggregate = Contains larger group masks")
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Refined legend with second-level clustering and overlap types
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan', linestyle='-',
               markersize=10, label=f'SC{smaller_group_id} - SMALLER group ({smaller_group_count} masks)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='magenta', linestyle='--',
               markersize=10, label=f'SC{larger_group_id} - LARGER group ({larger_group_count} masks)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
               markersize=10, label=f'Large masks - Simple overlaps ({simple_count})'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markersize=10, label=f'Large masks - Aggregate overlaps ({aggregate_count})'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=10, label='Large masks - No overlaps')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1),
             fontsize=10, framealpha=0.9)
    
    # Add refined explanation text box
    explanation_text = ("Refined Logic:\n"
                       f"• SC{smaller_group_id}(SMALLER) = Solid lines\n"
                       f"• SC{larger_group_id}(LARGER) = Dashed lines\n"
                       "• Simple = Only smaller group overlaps\n"
                       "• Aggregate = Contains larger group overlaps\n"
                       "• [S:n+L:m] = n smaller + m larger overlaps\n"
                       "• Thick red borders = Aggregate group")
    ax.text(0.02, 0.02, explanation_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save comprehensive visualization
    output_path = os.path.join(output_dir, f"comprehensive_overlap_analysis_{min_overlap_percentage}pct.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved enhanced comprehensive overlap visualization: {output_path}")
    
    # Refined summary statistics
    overlap_counts = [mask['overlap_count'] for mask in final_masks]
    smaller_overlaps = [mask.get('num_smaller_overlaps', 0) for mask in final_masks]
    larger_overlaps = [mask.get('num_larger_overlaps', 0) for mask in final_masks]
    
    print(f"Refined overlap summary:")
    print(f"  Large masks with 0 overlaps: {overlap_counts.count(0)}")
    print(f"  Large masks with simple overlaps (only smaller group): {simple_count}")
    print(f"  Large masks with aggregate overlaps (contains larger group): {aggregate_count}")
    print(f"  Maximum overlaps on a single large mask: {max(overlap_counts) if overlap_counts else 0}")
    print(f"  Average overlaps per large mask: {sum(overlap_counts)/len(overlap_counts):.1f}" if overlap_counts else 0)
    print(f"  Total smaller group overlaps: {sum(smaller_overlaps)}")
    print(f"  Total larger group overlaps: {sum(larger_overlaps)}")
    print(f"  Second-level clustering: SC{smaller_group_id}(SMALLER)={smaller_group_count}, SC{larger_group_id}(LARGER)={larger_group_count}")

def cluster_smaller_masks_kmeans(smaller_masks, n_clusters=2):
    """Perform second-level K-means clustering on smaller masks"""
    print(f"\n" + "="*60)
    print(f"SECOND-LEVEL CLUSTERING: Clustering smaller masks into {n_clusters} groups")
    print("="*60)
    
    if len(smaller_masks) < n_clusters:
        print(f"Not enough smaller masks ({len(smaller_masks)}) to cluster into {n_clusters} groups")
        return [smaller_masks] + [[] for _ in range(n_clusters - 1)], np.zeros(len(smaller_masks))
    
    # Extract features for smaller masks
    features = extract_mask_features(smaller_masks)
    
    # Apply log transformation to area to handle extreme values
    features_processed = features.copy()
    features_processed[:, 0] = np.log1p(features_processed[:, 0])  # log(area + 1)
    
    # Normalize features for better clustering
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_processed)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_normalized)
    
    # Check clustering balance
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Second-level clustering balance: {counts}")
    
    # Group masks by cluster
    clustered_smaller_masks = [[] for _ in range(n_clusters)]
    for i, label in enumerate(cluster_labels):
        clustered_smaller_masks[label].append(smaller_masks[i])
    
    # Print cluster statistics
    for i, cluster in enumerate(clustered_smaller_masks):
        if len(cluster) > 0:
            areas = [mask['area'] for mask in cluster]
            print(f"Small Cluster {i}: {len(cluster)} masks, "
                  f"area range: {min(areas):.0f}-{max(areas):.0f}, "
                  f"mean area: {np.mean(areas):.0f}")
        else:
            print(f"Small Cluster {i}: 0 masks")
    
    print(f"Second-level clustering completed: {len(smaller_masks)} → {[len(c) for c in clustered_smaller_masks]}")
    
    return clustered_smaller_masks, cluster_labels

def create_second_level_clustering_visualization(image, clustered_smaller_masks, smaller_cluster_labels, output_dir):
    """Create visualization showing the second-level clustering of smaller masks."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    print(f"\nCreating second-level clustering visualization...")
    
    if not clustered_smaller_masks or all(len(cluster) == 0 for cluster in clustered_smaller_masks):
        print("No smaller masks to visualize for second-level clustering.")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(image)
    
    # Colors for different small clusters
    cluster_colors = ['cyan', 'magenta', 'yellow', 'orange', 'lime']
    
    # Draw each cluster of smaller masks
    for cluster_idx, cluster_masks in enumerate(clustered_smaller_masks):
        if not cluster_masks:
            continue
            
        color = cluster_colors[cluster_idx % len(cluster_colors)]
        
        for mask_data in cluster_masks:
            bbox = mask_data['bbox']
            x, y, w, h = bbox
            
            # Draw bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Add mask ID label
            label_text = f"SC{cluster_idx}:{mask_data['original_sam_id']}"
            ax.text(x + w/2, y + h/2, label_text, fontsize=5, color='white', 
                   bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.9),
                   ha='center', va='center', weight='bold')
    
    # Create title and legend
    cluster_counts = [len(cluster) for cluster in clustered_smaller_masks]
    total_small_masks = sum(cluster_counts)
    
    title = f"Second-Level Clustering of Smaller Masks\nTotal: {total_small_masks} masks, Clusters: {cluster_counts}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend for clusters
    legend_elements = []
    for cluster_idx, count in enumerate(cluster_counts):
        if count > 0:
            color = cluster_colors[cluster_idx % len(cluster_colors)]
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                            markerfacecolor=color, markersize=10, 
                                            label=f'Small Cluster {cluster_idx} ({count} masks)'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(output_dir, "second_level_clustering_smaller_masks.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved second-level clustering visualization: {output_path}")

def validate_and_fix_bbox_data(masks, context_name=""):
    """Validate and fix bounding box data to prevent visualization issues"""
    print(f"Validating bbox data for {len(masks)} masks ({context_name})...")
    
    fixed_count = 0
    removed_count = 0
    valid_masks = []
    
    for i, mask_data in enumerate(masks):
        bbox = mask_data['bbox']
        
        # Convert bbox to list of integers if needed
        try:
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            
            # Ensure all values are integers
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Validate bbox values
            if w <= 0 or h <= 0:
                print(f"  Removing mask {i}: Invalid dimensions w={w}, h={h}")
                removed_count += 1
                continue
                
            if x < 0 or y < 0:
                print(f"  Removing mask {i}: Negative coordinates x={x}, y={y}")
                removed_count += 1
                continue
            
            # Update bbox in mask_data
            original_bbox = mask_data['bbox']
            mask_data['bbox'] = [x, y, w, h]
            
            if original_bbox != mask_data['bbox']:
                fixed_count += 1
            
            valid_masks.append(mask_data)
            
        except (ValueError, TypeError, IndexError) as e:
            print(f"  Removing mask {i}: Cannot process bbox {bbox}, error: {e}")
            removed_count += 1
            continue
    
    if fixed_count > 0:
        print(f"  Fixed {fixed_count} bbox data type issues")
    if removed_count > 0:
        print(f"  Removed {removed_count} masks with invalid bbox data")
    
    print(f"  Validated: {len(valid_masks)}/{len(masks)} masks are valid")
    return valid_masks

def main():
    # Configuration
    model_type = "vit_h"  # Using the base model (smaller, less GPU memory)
    crop_n_layers = 3
    
    # Paths
    image_path = "image/Test/GY_image.png"
    base_output_dir = "image/Output"
    
    # Create output directory with model and layer info
    output_folder_name = f"{model_type}_layer_{crop_n_layers}_second_clustering_enhanced_overlap_analysis_2"
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
    
    # NEW: Remove duplicate masks within each cluster (overlap > 60%)
    cleaned_clustered_masks, removed_duplicates_per_cluster = remove_duplicate_masks_within_clusters(
        clustered_masks, min_overlap_percentage=30.0
    )
    
    # Update clustered_masks to use the cleaned version
    clustered_masks = cleaned_clustered_masks
    
    # Validate bbox data before overlap analysis
    print("\nValidating bbox data before overlap analysis...")
    for i, cluster in enumerate(clustered_masks):
        clustered_masks[i] = validate_and_fix_bbox_data(cluster, f"cluster_{i}")
    
    # NEW: Analyze overlap between smaller and larger masks
    overlap_analysis_results = analyze_mask_overlaps(clustered_masks, image, output_dir)
    
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
    save_comprehensive_metadata(all_masks, filtered_masks, clustered_masks, cluster_labels, output_dir, removed_edge, removed_circularity, removed_multi_blob, removed_duplicates_per_cluster)
    
    # Save individual masks
    save_all_individual_masks(image, all_masks, filtered_masks, removed_edge, removed_circularity, removed_multi_blob, clustered_masks, output_dir, removed_duplicates_per_cluster)
    
    # Create removed duplicates visualization
    create_removed_duplicates_visualization(image, removed_duplicates_per_cluster, output_dir)
    
    # Create comprehensive overlap visualization
    create_comprehensive_overlap_visualization(overlap_analysis_results, image, output_dir, 60.0)
    
    # Create second-level clustering visualization
    create_second_level_clustering_visualization(image, overlap_analysis_results['clustered_smaller_masks'], overlap_analysis_results['smaller_cluster_labels'], output_dir)
    
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
    print(f"Duplicate removal: 60% bounding box overlap threshold")
    print(f"Cluster 0: {len(clustered_masks[0])} masks")
    print(f"Cluster 1: {len(clustered_masks[1])} masks")
    
    # Show duplicate removal statistics
    total_removed_duplicates = sum(len(removed) for removed in removed_duplicates_per_cluster)
    if total_removed_duplicates > 0:
        print(f"Removed duplicates: {total_removed_duplicates} masks")
        for i, removed in enumerate(removed_duplicates_per_cluster):
            if len(removed) > 0:
                print(f"  Cluster {i}: {len(removed)} duplicates removed")
    
    # Show overlap analysis results with refined logic
    if overlap_analysis_results and overlap_analysis_results['overlap_summary']:
        overlap_summary = overlap_analysis_results['overlap_summary']
        print(f"\nREFINED OVERLAP ANALYSIS RESULTS:")
        print(f"Large masks analyzed: {overlap_summary['total_large_masks']}")
        print(f"  Simple overlaps (only smaller group): {overlap_summary['simple_overlaps']}")
        print(f"  Aggregate overlaps (contains larger group): {overlap_summary['aggregate_overlaps']}")
        print(f"Minimum overlap required: {overlap_summary['min_overlap_percentage']}%")
        
        # Show refined logic breakdown
        if 'refined_logic' in overlap_summary:
            refined = overlap_summary['refined_logic']
            print(f"Refined logic breakdown:")
            print(f"  Total smaller group overlaps: {refined['total_smaller_group_overlaps']}")
            print(f"  Total larger group overlaps: {refined['total_larger_group_overlaps']}")
        
        # Show second-level clustering info with refined labeling
        if 'second_level_clustering' in overlap_summary:
            second_level = overlap_summary['second_level_clustering']
            print(f"Second-level clustering of smaller masks:")
            print(f"  Total smaller masks: {second_level['total_smaller_masks']}")
            print(f"  SC{second_level['smaller_group_id']} (SMALLER group): {second_level['smaller_clusters'][second_level['smaller_group_id']]} masks (avg area: {second_level['smaller_group_avg_area']:.0f})")
            print(f"  SC{second_level['larger_group_id']} (LARGER group): {second_level['smaller_clusters'][second_level['larger_group_id']]} masks (avg area: {second_level['larger_group_avg_area']:.0f})")
        
        print("Overlap distribution:")
        for overlap_count, frequency in overlap_summary['overlap_distribution'].items():
            print(f"  {frequency} large masks have {overlap_count} total overlapping masks")
    
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
    print("- Removed duplicate masks within clusters (overlap > 20%, removes larger masks)")
    print("\nOptimizations applied:")
    print(f"- Used {model_type} model for better segmentation")
    print(f"- Used crop_n_layers={crop_n_layers} for detailed segmentation")
    print("- K-means clustering with 6 features (including circularity)")
    print("- Post-clustering duplicate removal based on bounding box overlap")
    print("- Second-level K-means clustering on smaller masks")
    print("- Refined overlap logic: Simple (only smaller group) vs Aggregate (contains larger group)")
    print("- Enhanced visualizations with mask numbers and refined overlap analysis")
    print("- Comprehensive metadata generation with refined logic data")
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
    print("REFINED OVERLAP ANALYSIS:")
    print("- overlap_analysis_masks_80pct.png (large masks with refined overlap counts & aggregate groups)")
    print("- comprehensive_overlap_analysis_80pct.png (both small and large masks with refined overlap analysis)")
    print("- second_level_clustering_smaller_masks.png (second-level clustering showing smaller vs larger groups)")
    print("- overlap_analysis_details.csv (detailed overlap information with refined logic data)")
    print("DUPLICATE REMOVAL:")
    print("- removed_duplicate_masks.png (visualization of all removed duplicates)")
    print("INDIVIDUAL MASKS:")
    print("- all_masks_kmeans (all masks with kmeans clustering)")
    print("- cluster0 (cluster 0 masks after duplicate removal)")
    print("- cluster1 (cluster 1 masks after duplicate removal)")
    print("- filtered_edge (edge-touching masks)")
    print("- filtered_low_circularity (low circularity masks)")
    print("- filtered_multi_blob (multi-blob masks)")
    if any(len(removed) > 0 for removed in removed_duplicates_per_cluster):
        print("- filtered_duplicates_cluster0 (duplicates removed from cluster 0)")
        print("- filtered_duplicates_cluster1 (duplicates removed from cluster 1)")
    print("="*70)

if __name__ == "__main__":
    main() 