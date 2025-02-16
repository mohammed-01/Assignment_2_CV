import os
import cv2
import numpy as np
import random

# Paths
image_dir = "D:/LV-MHP-v2/LV-MHP-V2-Subset/train/images"  # Folder with original images
mask_dir = "D:/LV-MHP-v2/LV-MHP-V2-Subset/train/masks"  # Folder with generated masks
output_dir = "D:/LV-MHP-v2/LV-MHP-V2-Subset/train/overlayed_masks"  # Where to save overlays

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get a list of available images
image_files = os.listdir(image_dir)

# Loop through all images
for image_name in image_files:
    image_path = os.path.join(image_dir, image_name)
    mask_path = os.path.join(mask_dir, image_name.replace(".jpg", ".png"))  # Ensure mask has same name

    # Load image and mask
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Skipping {image_name}: Unable to load image.")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"⚠️ Skipping {image_name}: Unable to load mask.")
        continue

    # Generate random colors for instances
    unique_instances = np.unique(mask)
    instance_colors = {inst: np.random.randint(0, 255, (3,), dtype=np.uint8) for inst in unique_instances if inst > 0}

    # Create a color mask
    color_mask = np.zeros_like(image, dtype=np.uint8)
    for inst, color in instance_colors.items():
        color_mask[mask == inst] = color

    # Overlay mask on image
    overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

    # Save the result
    overlay_path = os.path.join(output_dir, image_name)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))  # Save as overlayed image

print("✅ Overlaid mask images saved in:", output_dir)