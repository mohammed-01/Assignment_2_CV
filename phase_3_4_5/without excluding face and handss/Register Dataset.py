import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

# Paths to dataset
BASE_PATH = "D:/LV-MHP-v2/LV-MHP-V2-Subset/"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")
TRAIN_ANNOTATIONS = os.path.join(TRAIN_PATH, "data_list.json")
TEST_ANNOTATIONS = os.path.join(TEST_PATH, "data_list.json")

def load_lv_mhp_v2_annotations(json_file):
    with open(json_file, "r") as f:
        dataset_dicts = json.load(f)

    # Map gender to category_id
    gender_to_category = {
        "male": 0,
        "female": 1,
    }

    for idx, item in enumerate(dataset_dicts):
        # Use "filepath" instead of "file_name"
        item["file_name"] = item["filepath"]

        # Add image_id (use the index or generate a unique ID)
        item["image_id"] = idx  # Use the index as the image_id

        # Convert annotations to Detectron2 format
        annotations = []
        for ann in item["bboxes"]:
            # Map gender to category_id
            gender = ann["gender"]
            category_id = gender_to_category.get(gender, -1)  # Default to -1 if gender is unknown

            if category_id == -1:
                print(f"Warning: Unknown gender '{gender}' in item {idx}. Skipping this annotation.")
                continue

            annotations.append({
                "bbox": [ann["x1"], ann["y1"], ann["x2"], ann["y2"]],  # Bounding box coordinates
                "bbox_mode": BoxMode.XYXY_ABS,  # Bounding box format
                "category_id": category_id,  # Use gender as the category_id
            })

        item["annotations"] = annotations
        del item["bboxes"]  # Remove the old key to avoid confusion

    return dataset_dicts

def register_lv_mhp_v2():
    if "lv_mhp_v2_train" not in DatasetCatalog.list():
        DatasetCatalog.register("lv_mhp_v2_train", lambda: load_lv_mhp_v2_annotations(TRAIN_ANNOTATIONS))
        MetadataCatalog.get("lv_mhp_v2_train").set(thing_classes=["male", "female"])  # Gender classes
        print("Successfully registered: lv_mhp_v2_train")

    if "lv_mhp_v2_test" not in DatasetCatalog.list():
        DatasetCatalog.register("lv_mhp_v2_test", lambda: load_lv_mhp_v2_annotations(TEST_ANNOTATIONS))
        MetadataCatalog.get("lv_mhp_v2_test").set(thing_classes=["male", "female"])  # Gender classes
        print("Successfully registered: lv_mhp_v2_test")

# Run registration when this file is executed
if __name__ == "__main__":
    register_lv_mhp_v2()
