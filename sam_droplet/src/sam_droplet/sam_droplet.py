"""Main module."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# Get the path to the project root (3 levels up from this script)
project_root = Path(__file__).parent.parent.parent
image_path = project_root / "images" / "ddPCR_0_0.png"

print(f"Project root: {project_root}")
print(f"Image path: {image_path}")
print(f"Image file exists: {image_path.exists()}")

if not image_path.exists():
    raise FileNotFoundError(f"Image file not found at: {image_path}")

image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Also fix the model path
model_path = project_root / "model" / "sam_vit_h_4b8939.pth"
sam_checkpoint = str(model_path)
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())

# Save each mask as an individual PNG file
output_dir = project_root / "output_masks"
output_dir.mkdir(exist_ok=True)

print(f"Saving {len(masks)} masks to: {output_dir}")

for i, mask in enumerate(masks):
    # Get the segmentation mask (boolean array)
    segmentation = mask['segmentation']
    
    # Convert boolean mask to uint8 (0 or 255)
    mask_image = (segmentation * 255).astype(np.uint8)
    
    # Save the mask
    mask_filename = output_dir / f"mask_{i:04d}.png"
    cv2.imwrite(str(mask_filename), mask_image)
    
    if i % 10 == 0:  # Print progress every 10 masks
        print(f"Saved mask {i+1}/{len(masks)}")

print(f"All masks saved successfully to {output_dir}")

