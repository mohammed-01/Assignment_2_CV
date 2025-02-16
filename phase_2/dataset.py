import os
import json
from pathlib import Path
from PIL import Image, ImageFile
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import numpy as np
import cv2
from pycocotools import mask as mask_utils

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set dataset paths
dataset_path = Path("D:/LV-MHP-v2/train")  # Update this path to your dataset location
train_json_path = Path(r"D:\LV-MHP-v2\train_set_annotations\data_list_fixed.json")
val_json_path = Path(r"D:\LV-MHP-v2\validation_set_annotations\data_list_fixed.json")

# Function to generate masks from parsing annotations (excluding hands and face)
def generate_mask(parsing_annotation_path, image_height, image_width):
    parsing_mask = cv2.imread(str(parsing_annotation_path), cv2.IMREAD_GRAYSCALE)
    if parsing_mask is None:
        raise FileNotFoundError(f"Unable to read parsing annotation file: {parsing_annotation_path}")

    parsing_mask = parsing_mask.astype(np.uint8)
    parsing_mask = cv2.resize(parsing_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

    binary_mask = np.zeros_like(parsing_mask, dtype=np.uint8)
    binary_mask[(parsing_mask != 4) & (parsing_mask != 13)] = 1  # Exclude hands (4) and face (13)

    if np.sum(binary_mask) == 0:
        print(f"Warning: Empty mask generated for {parsing_annotation_path}")

    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    return rle

# Function to convert JSON to Detectron2 format
def convert_json_to_detectron2(json_path, dataset_path):
    print("Loading dataset from:", json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    dataset_dicts = []
    for idx, entry in enumerate(data[:15000]):  # Limit to 5000 images
        try:
            image_path = dataset_path / "images" / entry["filepath"]
            if not image_path.exists():
                print(f"Warning: Image file not found: {image_path}")
                continue

            record = {
                "file_name": str(image_path),
                "image_id": idx,  # Ensure unique and consistent image_id
                "height": entry["height"],
                "width": entry["width"],
                "annotations": []
            }

            try:
                with open(record["file_name"], "rb") as f:
                    img = Image.open(f)
                    img.verify()  # Verify that the image is not truncated
                    img.close()
            except Exception as e:
                print(f"Warning: Skipping corrupted image {record['file_name']} - {e}")
                continue  # Skip this image and move to the next one

            for bbox in entry["bboxes"]:
                parsing_annotation_path = dataset_path / "parsing_annos" / bbox["ann_path"]
                if not parsing_annotation_path.exists():
                    print(f"Warning: Parsing annotation file not found: {parsing_annotation_path}")
                    continue

                mask = generate_mask(parsing_annotation_path, entry["height"], entry["width"])

                obj = {
                    "bbox": [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": mask,
                    "category_id": 0
                }
                record["annotations"].append(obj)

            dataset_dicts.append(record)
        except Exception as e:
            print(f"Error processing entry {idx}: {e}")
            continue

    print(f"Loaded {len(dataset_dicts)} entries from {json_path}")
    return dataset_dicts

# Register dataset
def register_datasets():
    if "lv_mhp_v2_train" not in DatasetCatalog.list():
        DatasetCatalog.register("lv_mhp_v2_train", lambda: convert_json_to_detectron2(train_json_path, dataset_path))
        MetadataCatalog.get("lv_mhp_v2_train").set(thing_classes=["person"])

    if "lv_mhp_v2_val" not in DatasetCatalog.list():
        DatasetCatalog.register("lv_mhp_v2_val", lambda: convert_json_to_detectron2(val_json_path, dataset_path))
        MetadataCatalog.get("lv_mhp_v2_val").set(thing_classes=["person"])

register_datasets()
print("✅ Training & Validation datasets registered successfully!")