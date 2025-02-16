import os
import json
import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from shapely.geometry import Polygon
from skimage import measure
import matplotlib.pyplot as plt  # Add this import

# Paths to dataset
BASE_PATH = "D:/LV-MHP-v2/LV-MHP-V2-Subset/"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")
TRAIN_ANNOTATIONS = os.path.join(TRAIN_PATH, "data_list.json")
TEST_ANNOTATIONS = os.path.join(TEST_PATH, "data_list.json")
TRAIN_MASK_PATH = os.path.join(TRAIN_PATH, "masks")  # Path to train masks
TEST_MASK_PATH = os.path.join(TEST_PATH, "masks")  # Path to test masks


def mask_to_polygons(mask):
    """
    Convert a binary mask to a list of polygons.
    """
    # Find contours in the binary mask
    contours = measure.find_contours(mask, 0.5)

    # Convert contours to polygons
    polygons = []
    for contour in contours:
        # Flip from (row, col) to (x, y)
        contour = np.flip(contour, axis=1)
        # Simplify polygon if necessary
        polygon = Polygon(contour)
        polygon = polygon.simplify(1.0, preserve_topology=False)
        if polygon.geom_type == 'Polygon':
            polygons.append(np.array(polygon.exterior.coords).ravel().tolist())
        elif polygon.geom_type == 'MultiPolygon':
            for poly in polygon.geoms:
                polygons.append(np.array(poly.exterior.coords).ravel().tolist())

    return polygons


def debug_mask_and_polygons(mask, polygons, image_path):
    """
    Debug function to visualize the mask and polygons.
    """
    plt.imshow(mask, cmap='gray')
    for polygon in polygons:
        polygon = np.array(polygon).reshape(-1, 2)
        plt.plot(polygon[:, 0], polygon[:, 1], 'r-', linewidth=2)
    plt.title(f"Mask and Polygons for {image_path}")
    #plt.show()


def load_lv_mhp_v2_annotations(json_file, mask_path):
    with open(json_file, "r") as f:
        dataset_dicts = json.load(f)

    # Map gender to category_id
    gender_to_category = {
        "male": 0,
        "female": 1,
    }

    valid_dataset_dicts = []  # Store valid annotations

    for idx, item in enumerate(dataset_dicts):
        # Use "filepath" instead of "file_name"
        item["file_name"] = item["filepath"]

        # Add image_id (use the index or generate a unique ID)
        item["image_id"] = idx  # Use the index as the image_id

        # Load the corresponding mask
        mask_file = os.path.join(mask_path, os.path.basename(item["file_name"]).replace(".jpg", ".png"))
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
        mask = (mask > 0).astype(np.uint8)  # Ensure mask is binary (0 or 1)

        # Check if the mask is empty
        if np.sum(mask) == 0:
            print(f"Warning: Empty mask for image {item['file_name']}. Skipping this image.")
            continue  # Skip this image if the mask is empty

        # Convert mask to polygons
        polygons = mask_to_polygons(mask)

        # Check if polygons are empty
        if len(polygons) == 0:
            print(f"Warning: No valid polygons for image {item['file_name']}. Skipping this image.")
            continue  # Skip this image if no valid polygons are found

        # Debug: Visualize the mask and polygons
        debug_mask_and_polygons(mask, polygons, item["file_name"])

        # Convert annotations to Detectron2 format
        annotations = []
        for ann in item["bboxes"]:
            # Map gender to category_id
            gender = ann["gender"]
            category_id = gender_to_category.get(gender, -1)  # Default to -1 if gender is unknown
            if category_id == -1:
                print(f"Warning: Unknown gender '{gender}' in item {idx}. Skipping this annotation.")
                continue

            # Create annotation dictionary
            annotations.append({
                "bbox": [ann["x1"], ann["y1"], ann["x2"], ann["y2"]],  # Bounding box coordinates
                "bbox_mode": BoxMode.XYXY_ABS,  # Bounding box format
                "category_id": category_id,  # Use gender as the category_id
                "segmentation": polygons,  # Add the polygons
            })

        item["annotations"] = annotations
        del item["bboxes"]  # Remove the old key to avoid confusion

        # Add valid annotations to the dataset
        valid_dataset_dicts.append(item)

    return valid_dataset_dicts


def register_lv_mhp_v2():
    if "lv_mhp_v2_train" not in DatasetCatalog.list():
        DatasetCatalog.register("lv_mhp_v2_train",
                                lambda: load_lv_mhp_v2_annotations(TRAIN_ANNOTATIONS, TRAIN_MASK_PATH))
        MetadataCatalog.get("lv_mhp_v2_train").set(thing_classes=["male", "female"])  # Gender classes
        print("Successfully registered: lv_mhp_v2_train")

    if "lv_mhp_v2_test" not in DatasetCatalog.list():
        DatasetCatalog.register("lv_mhp_v2_test", lambda: load_lv_mhp_v2_annotations(TEST_ANNOTATIONS, TEST_MASK_PATH))
        MetadataCatalog.get("lv_mhp_v2_test").set(thing_classes=["male", "female"])  # Gender classes
        print("Successfully registered: lv_mhp_v2_test")


# Run registration when this file is executed
if __name__ == "__main__":
    register_lv_mhp_v2()