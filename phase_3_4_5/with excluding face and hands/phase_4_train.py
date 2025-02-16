import os
import torch
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from phase4_register_dataset import register_lv_mhp_v2
from phase_4_config import get_train_config

class TrainerWithEvaluator(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create an evaluator for the given dataset.
        This evaluator will be used during training to evaluate the model on the validation set.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

if __name__ == "__main__":
    # Register dataset
    register_lv_mhp_v2()

    # Load config
    cfg = get_train_config()

    # Ensure CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is NOT available. Training on CPU.")

    # Trainer setup
    trainer = TrainerWithEvaluator(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()  # Start training