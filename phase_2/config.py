from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import transforms as T

def get_train_config():
    # Load the default Detectron2 configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Dataset configuration
    cfg.DATASETS.TRAIN = ("lv_mhp_v2_train",)
    cfg.DATASETS.TEST = ("lv_mhp_v2_val",)

    # Model weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # Number of classes (1 for person)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Batch size
    cfg.SOLVER.IMS_PER_BATCH = 1  # Adjust based on your GPU memory

    # Learning rate
    cfg.SOLVER.BASE_LR = 0.0025

    # Maximum iterations
    cfg.SOLVER.MAX_ITER = 4000  # Increased for proper training

    # Learning rate schedule steps
    cfg.SOLVER.STEPS = (2000, 3000)  # Adjust based on MAX_ITER
    cfg.SOLVER.GAMMA = 0.1  # Learning rate decay factor

    # Output directory
    cfg.OUTPUT_DIR = "./output"

    # DataLoader workers
    cfg.DATALOADER.NUM_WORKERS = 0  # Adjust based on your CPU cores

    # Mixed precision training
    cfg.SOLVER.AMP.ENABLED = False

    # Data augmentation
    cfg.DATALOADER.TRAIN_TRANSFORMS = [
        T.RandomFlip(prob=0.5),  # Random horizontal flip
        T.RandomBrightness(0.8, 1.2),  # Random brightness adjustment
        T.RandomContrast(0.8, 1.2),  # Random contrast adjustment
        T.RandomSaturation(0.8, 1.2),  # Random saturation adjustment
    ]

    # Anchor sizes for better handling of small/medium/large objects
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]  # Adjust anchor sizes

    # Non-maximum suppression (NMS) threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  # Lower threshold for tighter NMS

    # Evaluation settings
    cfg.TEST.EVAL_PERIOD = 0  # Evaluate every 500 iterations
    cfg.TEST.DETECTIONS_PER_IMAGE = 100  # Maximum number of detections per image

    return cfg