import os
import cv2
import numpy as np
import glob

# Paths
train_list_path = "D:/LV-MHP-v2/LV-MHP-V2-Subset/list/test.txt"  # List of 150 selected images
parsing_anno_path = "D:/LV-MHP-v2/val/parsing_annos"  # Parsing annotations folder
dest_mask_path = "D:/LV-MHP-v2/LV-MHP-V2-Subset/test/masks"  # Output mask folder

# Create output directory if it doesn't exist
os.makedirs(dest_mask_path, exist_ok=True)

# Classes to exclude (face, left hand, right hand)
EXCLUDED_CLASSES = {3, 7, 8}

# Load selected image filenames (without extensions)
with open(train_list_path, "r") as f:
    selected_images = set(line.strip().split('.')[0] for line in f.readlines())

# Process each selected image
for base_name in selected_images:
    mask = None  # Initialize empty mask

    # Find all annotation files matching this image base name
    annotation_files = glob.glob(os.path.join(parsing_anno_path, f"{base_name}_*.png"))

    if not annotation_files:
        print(f"⚠ No annotation found for {base_name}")
        continue

    for idx, anno_file in enumerate(annotation_files):
        # Load the annotation mask
        ann = cv2.imread(anno_file, cv2.IMREAD_GRAYSCALE)

        # Remove excluded classes (face, hands)
        ann_filtered = np.where(np.isin(ann, list(EXCLUDED_CLASSES)), 0, ann)

        # Merge instances into a single mask
        if mask is None:
            mask = (ann_filtered > 0).astype(np.uint8) * (idx + 1)
        else:
            mask = np.maximum(mask, (ann_filtered > 0).astype(np.uint8) * (idx + 1))

    # Save the final mask
    if mask is not None:
        cv2.imwrite(os.path.join(dest_mask_path, f"{base_name}.png"), mask * 255)
    else:
        print(f"⚠ No valid mask generated for {base_name}")

print("✅ Masks generated successfully for training images.")
