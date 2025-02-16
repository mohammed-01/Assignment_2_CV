import random
import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from phase4_register_dataset import register_lv_mhp_v2  # Import dataset registration

# Re-register dataset before using it
register_lv_mhp_v2()

# Load dataset metadata
metadata = MetadataCatalog.get("lv_mhp_v2_train")
dataset_dicts = DatasetCatalog.get("lv_mhp_v2_train")

# Pick a random sample image
sample = random.choice(dataset_dicts)

# Read and convert image for display
img = cv2.imread(sample["file_name"])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

# Visualize using Detectron2's Visualizer
visualizer = Visualizer(img, metadata=metadata, scale=1.0)
out = visualizer.draw_dataset_dict(sample)

# Show the image with annotations
plt.figure(figsize=(10, 10))
plt.imshow(out.get_image())
plt.axis("off")
plt.show()
