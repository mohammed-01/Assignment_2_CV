from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file, get_checkpoint_url

def get_train_config():
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Dataset settings
    cfg.DATASETS.TRAIN = ("lv_mhp_v2_train",)
    cfg.DATASETS.TEST = ("lv_mhp_v2_test",)
    cfg.DATALOADER.NUM_WORKERS = 0  # Disable multiprocessing

    # Model settings
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Two classes: male and female
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # Lower confidence threshold for evaluation
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Disable mask head (since you're not using masks)
    cfg.MODEL.MASK_ON = False

    # Training settings
    cfg.SOLVER.IMS_PER_BATCH = 2  # Adjust based on your GPU memory
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = []  # No learning rate decay

    # Evaluation settings
    cfg.TEST.EVAL_PERIOD = 500  # Evaluate every 500 iterations

    # Reset final layers to match the new number of classes
    cfg.MODEL.RESET_FINAL_LAYERS = True  # Add this line

    return cfg
