"""
Advanced mask analysis module for SAM droplet segmentation server.
Integrates comprehensive mask filtering, clustering, and analysis capabilities.
"""

import os
import io
import base64
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import existing SAM droplet utilities
import sys
sys.path.append(str(Path(__file__).parent / "src"))
from sam_droplet.filters import (
    calculate_circularity, 
    calculate_max_feret_diameter,
    analyze_mask_pixels,
    analyze_edge_proximity
)

class AdvancedMaskAnalyzer:
    """Advanced mask analysis with filtering, clustering, and visualization."""
    
    def __init__(self):
        self.original_masks = []
        self.filtered_masks = []
        self.removed_masks = {
            'edge': [],
            'circularity': [],
            'distant_blob': []
        }
        self.clustered_masks = []
        self.cluster_labels = []
        self.overlap_analysis = None
        
    def add_blob_distance_to_masks(self, masks: List[Dict]) -> List[Dict]:
        """Add blob distance information to masks."""
        for mask_data in masks:
            mask = mask_data['segmentation']
            blob_distance = self.calculate_blob_distance(mask)
            mask_data['blob_distance'] = blob_distance
        return masks
    
    def calculate_blob_distance(self, mask: np.ndarray) -> float:
        """Calculate maximum distance between connected components (blobs) in a mask."""
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
    
    def filter_by_edge_proximity(self, masks: List[Dict], image_shape: Tuple, edge_threshold: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """Filter out masks touching image edges."""
        height, width = image_shape[:2]
        filtered_masks = []
        removed_edge = []
        
        for mask_data in masks:
            bbox = mask_data['bbox']
            x, y, w, h = bbox
            
            # Check if bounding box is close to edges
            touching_left = x <= edge_threshold
            touching_right = (x + w) >= (width - edge_threshold)
            touching_top = y <= edge_threshold
            touching_bottom = (y + h) >= (height - edge_threshold)
            
            is_touching_edge = touching_left or touching_right or touching_top or touching_bottom
            
            if is_touching_edge:
                removed_edge.append(mask_data)
            else:
                filtered_masks.append(mask_data)
        
        return filtered_masks, removed_edge
    
    def filter_by_circularity(self, masks: List[Dict], min_circularity: float = 0.53) -> Tuple[List[Dict], List[Dict]]:
        """Filter masks by circularity threshold."""
        filtered_masks = []
        removed_circularity = []
        
        for mask_data in masks:
            circularity = mask_data.get('circularity', 0.0)
            
            if circularity >= min_circularity:
                filtered_masks.append(mask_data)
            else:
                removed_circularity.append(mask_data)
        
        return filtered_masks, removed_circularity
    
    def filter_by_blob_distance(self, masks: List[Dict], max_distance: float = 50) -> Tuple[List[Dict], List[Dict]]:
        """Filter out masks where blobs are separated by more than max_distance pixels."""
        filtered_masks = []
        removed_distant_blob = []
        
        for mask_data in masks:
            blob_distance = mask_data.get('blob_distance', 0.0)
            
            if blob_distance <= max_distance:
                filtered_masks.append(mask_data)
            else:
                removed_distant_blob.append(mask_data)
        
        return filtered_masks, removed_distant_blob
    
    def extract_mask_features(self, masks: List[Dict]) -> np.ndarray:
        """Extract features for K-means clustering."""
        features = []
        
        for mask_data in masks:
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
    
    def cluster_masks_kmeans(self, masks: List[Dict], n_clusters: int = 2) -> Tuple[List[List[Dict]], np.ndarray]:
        """Cluster masks into groups using K-means."""
        if len(masks) < n_clusters:
            # Not enough masks to cluster, return all in one group
            return [masks] + [[] for _ in range(n_clusters - 1)], np.zeros(len(masks))
        
        # Extract features
        features = self.extract_mask_features(masks)
        
        # Apply log transformation to area to handle extreme values
        features_processed = features.copy()
        features_processed[:, 0] = np.log1p(features_processed[:, 0])  # log(area + 1)
        
        # Normalize features for better clustering
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_processed)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_normalized)
        
        # Group masks by cluster
        clustered_masks = [[] for _ in range(n_clusters)]
        for i, label in enumerate(cluster_labels):
            clustered_masks[label].append(masks[i])
        
        return clustered_masks, cluster_labels
    
    def calculate_bbox_overlap_percentage(self, bbox1: List, bbox2: List) -> float:
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
    
    def analyze_mask_overlaps(self, clustered_masks: List[List[Dict]], min_overlap_percentage: float = 80.0) -> Dict[str, Any]:
        """
        Analyze overlaps between smaller masks and larger masks.
        Returns larger masks labeled with the number of overlapping smaller masks.
        """
        if len(clustered_masks) < 2:
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
            else:
                larger_masks = cluster_1
                smaller_masks = cluster_0
        else:
            larger_masks = cluster_0 if cluster_0 else cluster_1
            smaller_masks = cluster_1 if cluster_0 else cluster_0
        
        # Analyze each larger mask for overlaps with smaller masks
        final_masks = []
        overlap_details = []
        
        for i, large_mask in enumerate(larger_masks):
            large_bbox = large_mask['bbox']
            overlapping_small_masks = []
            overlap_percentages = []
            
            # Check overlap with each smaller mask
            for j, small_mask in enumerate(smaller_masks):
                small_bbox = small_mask['bbox']
                
                # Calculate overlap percentage using bounding boxes (what % of small bbox is covered by large bbox)
                overlap_pct = self.calculate_bbox_overlap_percentage(small_bbox, large_bbox)
                
                if overlap_pct >= min_overlap_percentage:
                    overlapping_small_masks.append({
                        'small_mask_id': small_mask['original_sam_id'],
                        'small_mask_index': j,
                        'overlap_percentage': overlap_pct,
                        'small_mask_area': small_mask['area']
                    })
                    overlap_percentages.append(overlap_pct)
            
            # Create final mask with overlap count
            num_overlaps = len(overlapping_small_masks)
            final_mask = large_mask.copy()
            final_mask['overlap_count'] = num_overlaps
            final_mask['overlapping_masks'] = overlapping_small_masks
            final_mask['overlap_label'] = f"{num_overlaps}"
            
            final_masks.append(final_mask)
            
            # Store detailed information
            overlap_detail = {
                'large_mask_id': large_mask['original_sam_id'],
                'large_mask_area': large_mask['area'],
                'overlap_count': num_overlaps,
                'overlapping_small_masks': overlapping_small_masks,
                'avg_overlap_percentage': np.mean(overlap_percentages) if overlap_percentages else 0.0
            }
            overlap_details.append(overlap_detail)
        
        # Create summary statistics
        overlap_counts = [mask['overlap_count'] for mask in final_masks]
        unique_counts, count_frequencies = np.unique(overlap_counts, return_counts=True)
        
        overlap_summary = {
            'total_large_masks': len(final_masks),
            'overlap_distribution': dict(zip(unique_counts.tolist(), count_frequencies.tolist())),
            'min_overlap_percentage': min_overlap_percentage,
            'overlap_details': overlap_details,
            'larger_cluster_avg_area': np.mean([mask['area'] for mask in larger_masks]) if larger_masks else 0,
            'smaller_cluster_avg_area': np.mean([mask['area'] for mask in smaller_masks]) if smaller_masks else 0
        }
        
        return {
            'final_masks': final_masks,
            'overlap_summary': overlap_summary,
            'larger_cluster_masks': larger_masks,
            'smaller_cluster_masks': smaller_masks
        }
    
    def create_comprehensive_analysis(self, 
                                    original_masks: List[Dict], 
                                    image_shape: Tuple,
                                    filtering_params: Dict = None) -> Dict[str, Any]:
        """
        Perform comprehensive mask analysis with filtering and clustering.
        
        Args:
            original_masks: List of original SAM masks
            image_shape: Shape of the original image (height, width, channels)
            filtering_params: Dictionary of filtering parameters
        
        Returns:
            Comprehensive analysis results
        """
        if filtering_params is None:
            filtering_params = {
                'edge_threshold': 5,
                'min_circularity': 0.53,
                'max_blob_distance': 50
            }
        
        # Store original masks with IDs
        for i, mask in enumerate(original_masks):
            mask['original_sam_id'] = i
            # Add circularity if not present
            if 'circularity' not in mask:
                mask['circularity'] = calculate_circularity(mask['segmentation'])
        
        # Add blob distance information
        self.original_masks = self.add_blob_distance_to_masks(original_masks)
        
        # Step 1: Filter by edge proximity
        edge_filtered_masks, removed_edge = self.filter_by_edge_proximity(
            self.original_masks, image_shape, filtering_params['edge_threshold']
        )
        
        # Step 2: Filter by circularity
        circularity_filtered_masks, removed_circularity = self.filter_by_circularity(
            edge_filtered_masks, filtering_params['min_circularity']
        )
        
        # Step 3: Filter by blob distance
        self.filtered_masks, removed_distant_blob = self.filter_by_blob_distance(
            circularity_filtered_masks, filtering_params['max_blob_distance']
        )
        
        # Store removed masks
        self.removed_masks = {
            'edge': removed_edge,
            'circularity': removed_circularity,
            'distant_blob': removed_distant_blob
        }
        
        # Step 4: Cluster filtered masks
        self.clustered_masks, self.cluster_labels = self.cluster_masks_kmeans(
            self.filtered_masks, n_clusters=2
        )
        
        # Step 5: Analyze overlaps between clusters
        overlap_min_percentage = filtering_params.get('overlap_min_percentage', 80.0)
        self.overlap_analysis = self.analyze_mask_overlaps(
            self.clustered_masks, overlap_min_percentage
        )
        
        # Calculate statistics
        analysis_results = {
            'processing_summary': {
                'original_masks': len(self.original_masks),
                'after_edge_filtering': len(edge_filtered_masks),
                'after_circularity_filtering': len(circularity_filtered_masks),
                'after_blob_filtering': len(self.filtered_masks),
                'removed_edge_masks': len(removed_edge),
                'removed_low_circularity_masks': len(removed_circularity),
                'removed_distant_blob_masks': len(removed_distant_blob),
                'final_clustered_masks': len(self.filtered_masks)
            },
            'filtering_criteria': filtering_params,
            'clustering_info': {
                'method': 'K-means',
                'n_clusters': 2,
                'cluster_sizes': [len(cluster) for cluster in self.clustered_masks]
            },
            'overlap_analysis': self.overlap_analysis['overlap_summary'] if self.overlap_analysis else None,
            'cluster_statistics': []
        }
        
        # Add cluster statistics
        for i, cluster in enumerate(self.clustered_masks):
            if cluster:
                areas = [mask['area'] for mask in cluster]
                circularities = [mask.get('circularity', 0.0) for mask in cluster]
                blob_distances = [mask.get('blob_distance', 0.0) for mask in cluster]
                
                cluster_stats = {
                    'cluster_id': i,
                    'mask_count': len(cluster),
                    'area_stats': {
                        'min': float(min(areas)),
                        'max': float(max(areas)),
                        'mean': float(np.mean(areas)),
                        'std': float(np.std(areas))
                    },
                    'circularity_stats': {
                        'min': float(min(circularities)),
                        'max': float(max(circularities)),
                        'mean': float(np.mean(circularities)),
                        'std': float(np.std(circularities))
                    },
                    'blob_distance_stats': {
                        'min': float(min(blob_distances)),
                        'max': float(max(blob_distances)),
                        'mean': float(np.mean(blob_distances)),
                        'std': float(np.std(blob_distances))
                    }
                }
                analysis_results['cluster_statistics'].append(cluster_stats)
        
        return analysis_results
    
    def create_summary_csv_data(self) -> List[Dict]:
        """Create CSV-ready data for all masks."""
        csv_data = []
        
        # Create sets for faster lookup
        removed_edge_set = set(id(mask) for mask in self.removed_masks['edge'])
        removed_circularity_set = set(id(mask) for mask in self.removed_masks['circularity'])
        removed_distant_blob_set = set(id(mask) for mask in self.removed_masks['distant_blob'])
        filtered_masks_set = set(id(mask) for mask in self.filtered_masks)
        
        # Find cluster assignments
        mask_to_cluster = {}
        for cluster_id, cluster in enumerate(self.clustered_masks):
            for mask in cluster:
                mask_to_cluster[id(mask)] = cluster_id
        
        # Find overlap information
        mask_to_overlap_info = {}
        if self.overlap_analysis and self.overlap_analysis['final_masks']:
            for final_mask in self.overlap_analysis['final_masks']:
                mask_to_overlap_info[id(final_mask)] = {
                    'overlap_count': final_mask['overlap_count'],
                    'overlapping_mask_ids': [o['small_mask_id'] for o in final_mask['overlapping_masks']]
                }
        
        for mask_data in self.original_masks:
            bbox = mask_data['bbox']
            cluster_id = mask_to_cluster.get(id(mask_data), -1)
            
            # Get overlap information
            overlap_info = mask_to_overlap_info.get(id(mask_data), {})
            overlap_count = overlap_info.get('overlap_count', 0)
            overlapping_ids = overlap_info.get('overlapping_mask_ids', [])
            
            row = {
                'mask_id': mask_data['original_sam_id'],
                'area': mask_data['area'],
                'circularity': mask_data.get('circularity', 0.0),
                'blob_distance': mask_data.get('blob_distance', 0.0),
                'stability_score': mask_data['stability_score'],
                'bbox_x': bbox[0],
                'bbox_y': bbox[1],
                'bbox_width': bbox[2],
                'bbox_height': bbox[3],
                'aspect_ratio': bbox[2] / bbox[3] if bbox[3] > 0 else 1.0,
                'cluster_id': cluster_id,
                'edge_touching': id(mask_data) in removed_edge_set,
                'low_circularity': id(mask_data) in removed_circularity_set,
                'distant_blob': id(mask_data) in removed_distant_blob_set,
                'included_in_clustering': id(mask_data) in filtered_masks_set,
                'overlap_count': overlap_count,
                'overlapping_small_mask_ids': ';'.join(map(str, overlapping_ids)) if overlapping_ids else ''
            }
            csv_data.append(row)
        
        return csv_data
    
    def create_visualization_summary(self, image: np.ndarray) -> Dict[str, str]:
        """Create base64 encoded visualization images."""
        visualizations = {}
        
        # Create overview visualization showing all filtered masks with clusters
        if self.filtered_masks:
            overview_img = self._create_cluster_visualization(image, self.filtered_masks, self.clustered_masks)
            visualizations['cluster_overview'] = self._numpy_to_base64(overview_img)
        
        # Create filtered mask visualizations
        for category, masks in self.removed_masks.items():
            if masks:
                viz_img = self._create_filtered_visualization(image, masks, category)
                visualizations[f'filtered_{category}'] = self._numpy_to_base64(viz_img)
        
        # Create overlap visualization
        if self.overlap_analysis and self.overlap_analysis['final_masks']:
            overlap_img = self._create_overlap_visualization(image, self.overlap_analysis)
            visualizations['overlap_analysis'] = self._numpy_to_base64(overlap_img)
        
        return visualizations
    
    def _create_cluster_visualization(self, image: np.ndarray, filtered_masks: List[Dict], clustered_masks: List[List[Dict]]) -> np.ndarray:
        """Create visualization showing clustered masks."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Color schemes for clusters
        colors = [plt.cm.Blues(0.7), plt.cm.Reds(0.7)]
        
        for cluster_id, cluster in enumerate(clustered_masks):
            color = colors[cluster_id % len(colors)]
            
            for mask_data in cluster:
                bbox = mask_data['bbox']
                x, y, w, h = bbox
                
                # Draw bounding box
                rect = patches.Rectangle((x, y), w, h, linewidth=1.5, 
                                       edgecolor=color[:3], facecolor='none')
                ax.add_patch(rect)
                
                # Add mask ID label
                ax.text(x, y - 2, str(mask_data['original_sam_id']), 
                       fontsize=6, color='white', 
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7),
                       ha='left', va='bottom', weight='bold')
        
        ax.set_title(f'Clustered Masks (Cluster 0: {len(clustered_masks[0])}, Cluster 1: {len(clustered_masks[1] if len(clustered_masks) > 1 else [])})')
        ax.axis('off')
        
        # Convert plot to numpy array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.close()
        
        return img_rgb
    
    def _create_filtered_visualization(self, image: np.ndarray, filtered_masks: List[Dict], category: str) -> np.ndarray:
        """Create visualization for filtered out masks."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Color schemes for different filter categories
        color_map = {
            'edge': plt.cm.Blues(0.7),
            'circularity': plt.cm.Reds(0.7),
            'distant_blob': plt.cm.Oranges(0.7)
        }
        color = color_map.get(category, plt.cm.Grays(0.7))
        
        for mask_data in filtered_masks:
            bbox = mask_data['bbox']
            x, y, w, h = bbox
            
            # Draw bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=1.5, 
                                   edgecolor=color[:3], facecolor='none')
            ax.add_patch(rect)
            
            # Add mask ID label
            ax.text(x, y - 2, str(mask_data['original_sam_id']), 
                   fontsize=6, color='white', 
                   bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7),
                   ha='left', va='bottom', weight='bold')
        
        ax.set_title(f'Filtered Out: {category.title()} Masks ({len(filtered_masks)} masks)')
        ax.axis('off')
        
        # Convert plot to numpy array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.close()
        
        return img_rgb
    
    def _create_overlap_visualization(self, image: np.ndarray, overlap_analysis: Dict) -> np.ndarray:
        """Create visualization showing larger masks labeled with overlap counts."""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(image)
        
        final_masks = overlap_analysis['final_masks']
        overlap_summary = overlap_analysis['overlap_summary']
        
        # Create color map for different overlap counts
        max_overlap = max([mask['overlap_count'] for mask in final_masks]) if final_masks else 0
        colors = plt.cm.viridis(np.linspace(0, 1, max_overlap + 1))
        
        # Draw each final mask with its overlap count label
        for mask_data in final_masks:
            bbox = mask_data['bbox']
            overlap_count = mask_data['overlap_count']
            x, y, w, h = bbox
            
            # Color based on overlap count
            color = colors[overlap_count] if overlap_count <= max_overlap else colors[-1]
            
            # Draw bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor=color[:3], facecolor='none')
            ax.add_patch(rect)
            
            # Add overlap count label
            label_text = f"ID:{mask_data['original_sam_id']}\n{overlap_count} overlaps"
            ax.text(x, y - 5, label_text, fontsize=8, color='white', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8),
                   ha='left', va='bottom', weight='bold')
        
        # Create title
        min_pct = overlap_summary.get('min_overlap_percentage', 80)
        title = f"Large Masks with Overlap Counts\n(Min {min_pct}% overlap required)"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add color bar legend
        if max_overlap > 0:
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=max_overlap))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Number of Overlapping Small Masks', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        # Convert plot to numpy array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.close()
        
        return img_rgb
    
    def _numpy_to_base64(self, img_array: np.ndarray) -> str:
        """Convert numpy array to base64 string."""
        pil_image = Image.fromarray(img_array)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"


# Global analyzer instance
mask_analyzer = AdvancedMaskAnalyzer() 