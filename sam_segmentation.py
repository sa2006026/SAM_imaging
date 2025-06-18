import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

def download_sam_model():
    """Download SAM model if not present"""
    import urllib.request
    
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    model_path = "sam_vit_b_01ec64.pth"
    
    if not os.path.exists(model_path):
        print("Downloading SAM model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully!")
    
    return model_path

def load_image(image_path):
    """Load and convert image to RGB"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def generate_masks(image, model_type="vit_b", crop_n_layers=2):
    """Generate masks using SAM"""
    # Download model if needed
    model_path = download_sam_model()
    
    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    
    # Create mask generator with specified parameters
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,  # Require a minimum region area
    )
    
    # Generate masks
    print("Generating masks...")
    masks = mask_generator.generate(image)
    print(f"Generated {len(masks)} masks")
    
    return masks

def save_masks(masks, output_dir, image_shape):
    """Save individual masks as PNG files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        
        # Convert boolean mask to uint8
        mask_img = (mask * 255).astype(np.uint8)
        
        # Save mask
        mask_path = os.path.join(output_dir, f"mask_{i:03d}.png")
        cv2.imwrite(mask_path, mask_img)
    
    print(f"Saved {len(masks)} masks to {output_dir}")

def create_interactive_visualization(image, masks, output_path="interactive_masks.html"):
    """Create interactive visualization with hover highlighting"""
    
    # Create figure
    fig = go.Figure()
    
    # Add original image as background
    fig.add_trace(go.Image(z=image, name="Original Image"))
    
    # Prepare mask overlays with robust color handling
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    colors = (base_colors * (len(masks) // len(base_colors) + 1))[:len(masks)]
    
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        bbox = mask_data['bbox']  # [x, y, width, height]
        
        # Create colored mask overlay
        colored_mask = np.zeros((*mask.shape, 4))  # RGBA
        # Convert hex color to RGB
        hex_color = colors[i % len(colors)]
        color_rgb = mcolors.hex2color(hex_color)
        colored_mask[mask] = [color_rgb[0], color_rgb[1], color_rgb[2], 0.6]  # Semi-transparent
        
        # Add mask as image trace
        fig.add_trace(go.Image(
            z=colored_mask,
            name=f"Mask {i}",
            visible=False,  # Initially hidden
            hovertemplate=f"Mask {i}<br>Area: {mask_data['area']}<br>Stability Score: {mask_data['stability_score']:.3f}<extra></extra>"
        ))
        
        # Add bounding box
        x, y, w, h = bbox
        fig.add_trace(go.Scatter(
            x=[x, x+w, x+w, x, x],
            y=[y, y, y+h, y+h, y],
            mode='lines',
            line=dict(color=hex_color, width=2),
            name=f"BBox {i}",
            visible=False,  # Initially hidden
            showlegend=False,
            hovertemplate=f"Mask {i}<br>Bbox: ({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f})<br>Area: {mask_data['area']}<br>Stability Score: {mask_data['stability_score']:.3f}<extra></extra>"
        ))
    
    # Create buttons for toggling masks
    buttons = []
    
    # Show all button
    visibility_all = [True] + [True] * (len(masks) * 2)  # Image + all masks and bboxes
    buttons.append(dict(
        label="Show All",
        method="update",
        args=[{"visible": visibility_all}]
    ))
    
    # Hide all button
    visibility_none = [True] + [False] * (len(masks) * 2)  # Only image visible
    buttons.append(dict(
        label="Hide All",
        method="update",
        args=[{"visible": visibility_none}]
    ))
    
    # Individual mask buttons (limit to first 20 for UI clarity)
    for i in range(min(20, len(masks))):
        visibility = [True] + [False] * (len(masks) * 2)  # Only image visible
        visibility[1 + i * 2] = True      # Show mask
        visibility[1 + i * 2 + 1] = True  # Show bbox
        
        buttons.append(dict(
            label=f"Mask {i}",
            method="update",
            args=[{"visible": visibility}]
        ))
    
    # Update layout
    fig.update_layout(
        title="Interactive SAM Segmentation Results",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            buttons=buttons,
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.01,
            xanchor="left",
            y=1.02,
            yanchor="top"
        )],
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Save interactive plot
    fig.write_html(output_path)
    print(f"Interactive visualization saved to {output_path}")
    
    return fig

def create_static_visualization(image, masks, output_path="masks_overview.png"):
    """Create static visualization showing all masks"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # All masks overlay
    overlay = image.copy()
    colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
    
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        color = colors[i][:3]  # RGB only
        overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 255 * 0.5
    
    axes[0, 1].imshow(overlay.astype(np.uint8))
    axes[0, 1].set_title(f"All Masks Overlay ({len(masks)} masks)")
    axes[0, 1].axis('off')
    
    # Masks with bounding boxes
    axes[1, 0].imshow(image)
    for i, mask_data in enumerate(masks):
        bbox = mask_data['bbox']
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor=colors[i], facecolor='none')
        axes[1, 0].add_patch(rect)
    axes[1, 0].set_title("Bounding Boxes")
    axes[1, 0].axis('off')
    
    # Mask statistics
    areas = [mask['area'] for mask in masks]
    stability_scores = [mask['stability_score'] for mask in masks]
    
    axes[1, 1].scatter(areas, stability_scores, alpha=0.6, c=range(len(masks)), cmap='Set3')
    axes[1, 1].set_xlabel('Mask Area (pixels)')
    axes[1, 1].set_ylabel('Stability Score')
    axes[1, 1].set_title('Mask Quality Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Static visualization saved to {output_path}")

def create_hover_visualization(image, masks, output_path="hover_masks.html"):
    """Create interactive visualization with hover highlighting for individual masks"""
    
    # Create figure with subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Original Image with Bounding Boxes", "Hover to Highlight Masks"),
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Left subplot: Original image with all bounding boxes
    fig.add_trace(go.Image(z=image, name="Original"), row=1, col=1)
    
    # Right subplot: Image for mask highlighting
    fig.add_trace(go.Image(z=image, name="Base Image"), row=1, col=2)
    
    # Add bounding boxes to left subplot
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    
    for i, mask_data in enumerate(masks):
        bbox = mask_data['bbox']
        x, y, w, h = bbox
        color = base_colors[i % len(base_colors)]
        
        # Add bounding box to left subplot
        fig.add_trace(go.Scatter(
            x=[x, x+w, x+w, x, x],
            y=[y, y, y+h, y+h, y],
            mode='lines',
            line=dict(color=color, width=2),
            name=f"Mask {i}",
            showlegend=False,
            hovertemplate=f"<b>Mask {i}</b><br>" +
                         f"Area: {mask_data['area']:,} pixels<br>" +
                         f"Bbox: ({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f})<br>" +
                         f"Stability: {mask_data['stability_score']:.3f}<br>" +
                         "<extra></extra>",
            hoveron='points+fills'
        ), row=1, col=1)
        
        # Add invisible scatter points for hover detection on right subplot
        center_x, center_y = x + w/2, y + h/2
        fig.add_trace(go.Scatter(
            x=[center_x],
            y=[center_y],
            mode='markers',
            marker=dict(size=max(w, h)/10, opacity=0.01, color=color),
            name=f"Hover {i}",
            showlegend=False,
            hovertemplate=f"<b>Mask {i}</b><br>" +
                         f"Area: {mask_data['area']:,} pixels<br>" +
                         f"Stability: {mask_data['stability_score']:.3f}<br>" +
                         "Hover to highlight mask<br>" +
                         "<extra></extra>",
            customdata=[i]  # Store mask index for JavaScript callback
        ), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title="Interactive SAM Segmentation - Hover to Highlight",
        height=600,
        showlegend=False,
        hovermode='closest'
    )
    
    # Update axes for both subplots
    for col in [1, 2]:
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=col)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, 
                        scaleanchor=f"x{col if col > 1 else ''}", row=1, col=col)
    
    # Add JavaScript for dynamic mask highlighting
    fig.write_html(output_path, include_plotlyjs=True)
    
    # Add custom JavaScript for mask highlighting
    with open(output_path, 'r') as f:
        html_content = f.read()
    
    # Insert custom CSS and JavaScript
    custom_script = """
    <style>
    .hover-mask { opacity: 0.7; transition: opacity 0.2s; }
    </style>
    <script>
    // This would require more complex implementation for true hover highlighting
    // For now, the hover tooltips provide the interaction
    </script>
    """
    
    html_content = html_content.replace('</head>', custom_script + '</head>')
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Hover visualization saved to {output_path}")
    return fig

def main():
    # Paths
    image_path = "image/Test/GY_image.png"
    output_dir = "image/Output"
    
    # Load image
    print("Loading image...")
    image = load_image(image_path)
    print(f"Image shape: {image.shape}")
    
    # Generate masks using SAM with specified parameters
    masks = generate_masks(image, model_type="vit_b", crop_n_layers=2)
    
    # Save individual masks
    masks_dir = os.path.join(output_dir, "masks")
    save_masks(masks, masks_dir, image.shape)
    
    # Create visualizations
    hover_path = os.path.join(output_dir, "hover_masks.html")
    static_path = os.path.join(output_dir, "masks_overview.png")
    
    create_hover_visualization(image, masks, hover_path)
    create_static_visualization(image, masks, static_path)
    
    # Save mask metadata
    metadata = []
    for i, mask_data in enumerate(masks):
        metadata.append({
            'mask_id': i,
            'area': int(mask_data['area']),
            'bbox': [int(x) for x in mask_data['bbox']],
            'stability_score': float(mask_data['stability_score']),
            'crop_box': [int(x) for x in mask_data['crop_box']] if 'crop_box' in mask_data else None
        })
    
    metadata_path = os.path.join(output_dir, "masks_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSegmentation complete!")
    print(f"- {len(masks)} masks saved to: {masks_dir}")
    print(f"- Hover visualization: {hover_path}")
    print(f"- Static overview: {static_path}")
    print(f"- Metadata: {metadata_path}")
    
    # Also create a simple mask viewer
    create_simple_mask_viewer(image, masks, os.path.join(output_dir, "simple_viewer.html"))

def create_simple_mask_viewer(image, masks, output_path):
    """Create a simple mask viewer with dropdown selection"""
    
    fig = go.Figure()
    
    # Add original image
    fig.add_trace(go.Image(z=image, name="Original Image"))
    
    # Add each mask as a separate trace (initially hidden)
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        bbox = mask_data['bbox']
        
        # Create colored mask overlay
        colored_mask = np.zeros((*mask.shape, 4))
        color = base_colors[i % len(base_colors)]
        color_rgb = mcolors.hex2color(color)
        colored_mask[mask] = [color_rgb[0], color_rgb[1], color_rgb[2], 0.6]
        
        fig.add_trace(go.Image(
            z=colored_mask,
            name=f"Mask {i}",
            visible=False
        ))
        
        # Add bounding box
        x, y, w, h = bbox
        fig.add_trace(go.Scatter(
            x=[x, x+w, x+w, x, x],
            y=[y, y, y+h, y+h, y],
            mode='lines',
            line=dict(color=color, width=3),
            name=f"BBox {i}",
            visible=False,
            showlegend=False
        ))
    
    # Create dropdown menu
    dropdown_buttons = []
    
    # Show original only
    visibility_original = [True] + [False] * (len(masks) * 2)
    dropdown_buttons.append(dict(
        label="Original Image",
        method="update",
        args=[{"visible": visibility_original}]
    ))
    
    # Individual mask options (show first 50 to avoid UI clutter)
    for i in range(min(50, len(masks))):
        visibility = [True] + [False] * (len(masks) * 2)
        visibility[1 + i * 2] = True      # Show mask
        visibility[1 + i * 2 + 1] = True  # Show bbox
        
        mask_data = masks[i]
        dropdown_buttons.append(dict(
            label=f"Mask {i} (Area: {mask_data['area']:,})",
            method="update",
            args=[{"visible": visibility}]
        ))
    
    fig.update_layout(
        title="SAM Mask Viewer - Select mask from dropdown",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        updatemenus=[dict(
            buttons=dropdown_buttons,
            direction="down",
            showactive=True,
            x=0.01,
            xanchor="left",
            y=0.99,
            yanchor="top"
        )],
        width=900,
        height=700
    )
    
    fig.write_html(output_path)
    print(f"Simple mask viewer saved to {output_path}")

if __name__ == "__main__":
    main() 