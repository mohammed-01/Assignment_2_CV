import tracemalloc
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from config import get_train_config
from dataset import register_datasets
import torch
import multiprocessing
from torch.cuda.amp import autocast, GradScaler
from detectron2.utils.logger import setup_logger
import logging

# Start memory monitoring
tracemalloc.start()

# Register datasets (only if not already registered)
register_datasets()

# Load config
cfg = get_train_config()

# Ensure Detectron2 uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.MODEL.DEVICE = str(device)  # Set the device in Detectron2 config

# Set up logger
setup_logger()
logger = logging.getLogger("detectron2")

# Log training details
logger.info("Starting training...")
logger.info(f"Using device: {cfg.MODEL.DEVICE}")
logger.info(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
logger.info(f"Learning rate: {cfg.SOLVER.BASE_LR}")

# Add evaluator for validation
evaluator = COCOEvaluator(
    dataset_name="lv_mhp_v2_val",
    output_dir=cfg.OUTPUT_DIR,
    tasks=("bbox",),  # Only evaluate bounding boxes
    distributed=False,
)

# Train the model
if __name__ == '__main__':
    # Fix for multiprocessing on Windows
    multiprocessing.freeze_support()

    # Initialize GradScaler for mixed precision training
    grad_scaler = GradScaler()  # Older initialization for compatibility

    # Create the trainer
    class TrainerWithEvaluator(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    trainer = TrainerWithEvaluator(cfg)
    trainer.model.to(device)  # Explicitly move the model to GPU/CPU

    # Override the run_step method to use the updated autocast API
    def run_step(self):
        """
        Override the run_step method to use the updated autocast API.
        """
        assert self.model.training, "Model is not in training mode!"

        # Initialize the data loader iterator if it doesn't exist
        if not hasattr(self, "_data_loader_iter"):
            self._data_loader_iter = iter(self.data_loader)

        try:
            # Fetch the next batch of data
            data = next(self._data_loader_iter)

            # Perform forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # Updated autocast API
                loss_dict = self.model(data)
                losses = sum(loss_dict.values())

            # Perform backward pass and update weights
            self.optimizer.zero_grad()
            grad_scaler.scale(losses).backward()  # Use GradScaler
            grad_scaler.step(self.optimizer)  # Use GradScaler
            grad_scaler.update()  # Use GradScaler

        except Exception as e:
            print(f"Error during training step: {e}")
            raise  # Re-raise the exception to stop training

    # Replace the original run_step method with the updated one
    trainer.run_step = run_step.__get__(trainer)

    # Start training
    try:
        trainer.resume_or_load(resume=False)
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Evaluate the model after training
    try:
        val_loader = build_detection_test_loader(cfg, "lv_mhp_v2_val")
        metrics = trainer.test(cfg, trainer.model, evaluators=evaluator)
        print("Evaluation Results:", metrics)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    # Print memory usage statistics
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    for stat in top_stats[:10]:
        print(stat)