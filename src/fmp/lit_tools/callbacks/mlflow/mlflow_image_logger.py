"""
Simple MLFlow image logging callback for PyTorch Lightning.

This module provides a simple callback for logging training images to MLFlow.
"""

from typing import Any, List, Optional

import lightning as L
import torch
import torchvision.transforms.functional as TF
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger

__all__ = ["MLFlowImageLogger"]


class MLFlowImageLogger(Callback):
    """
    Simple callback for logging training images to MLFlow.

    This callback logs sample images from training batches to MLFlow,
    providing visual insight into the training data.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 5,
        max_images_per_batch: int = 8,
        denormalize: bool = True,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
        artifact_path: str = "training_images",
        enable_image_grid: bool = False,
    ):
        """
        Args:
            log_every_n_epochs: Log images every N epochs
            max_images_per_batch: Maximum number of images to log per batch
            denormalize: Whether to denormalize images for logging
            normalize_mean: Mean values used for normalization (for denormalization)
            normalize_std: Std values used for normalization (for denormalization)
            artifact_path: Base path within MLFlow artifacts for images
            enable_image_grid: Enable Image Grid visualization (causes URL encoding bug in MLFlow)
        """
        super().__init__()

        self.log_every_n_epochs = log_every_n_epochs
        self.max_images_per_batch = max_images_per_batch
        self.denormalize = denormalize
        self.normalize_mean = normalize_mean or [
            0.485,
            0.456,
            0.406,
        ]  # ImageNet defaults
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]  # ImageNet defaults
        self.artifact_path = artifact_path
        self.enable_image_grid = enable_image_grid

        # Internal state
        self._last_logged_epoch = -1

    def _get_mlflow_logger(self, trainer: L.Trainer) -> Optional[MLFlowLogger]:
        """Get MLFlow logger from trainer."""
        if trainer.logger is None:
            return None

        if isinstance(trainer.logger, MLFlowLogger):
            return trainer.logger

        # Check if logger is a list/tuple and contains MLFlow logger
        if hasattr(trainer.logger, "__iter__"):
            for logger in trainer.logger:
                if isinstance(logger, MLFlowLogger):
                    return logger

        return None

    def _should_log(self, trainer: L.Trainer) -> bool:
        """Determine if images should be logged this epoch."""
        if not trainer.is_global_zero:
            return False

        # Always log during debug scenarios (very limited batches/epochs)
        if (
            (
                hasattr(trainer, "limit_train_batches")
                and trainer.limit_train_batches is not None
                and trainer.limit_train_batches <= 5
            )
            or (
                hasattr(trainer, "max_epochs")
                and trainer.max_epochs is not None
                and trainer.max_epochs <= 2
            )
            or (
                hasattr(trainer, "max_steps")
                and trainer.max_steps is not None
                and trainer.max_steps <= 5
            )
        ):
            return True

        current_epoch = trainer.current_epoch
        if current_epoch != self._last_logged_epoch:
            if current_epoch % self.log_every_n_epochs == 0:
                return True

        return False

    def _denormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Denormalize image tensor."""
        if not self.denormalize:
            return image

        # Convert lists to tensors
        mean = torch.tensor(self.normalize_mean).view(-1, 1, 1)
        std = torch.tensor(self.normalize_std).view(-1, 1, 1)

        # Move to same device
        mean = mean.to(image.device)
        std = std.to(image.device)

        # Denormalize
        denorm_image = image * std + mean

        # Clamp to valid range
        return torch.clamp(denorm_image, 0, 1)

    def _log_images_to_mlflow(self, images: torch.Tensor, trainer: L.Trainer):
        """Log images to MLFlow with configurable Image Grid support."""
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        # Take first N images from batch
        n_images = min(self.max_images_per_batch, images.shape[0])
        sample_images = images[:n_images]

        logged_count = 0
        epoch = trainer.current_epoch
        step = trainer.global_step

        for i, image in enumerate(sample_images):
            try:
                # Denormalize if needed
                if self.denormalize:
                    image = self._denormalize_image(image)

                # Ensure image is in [0, 1] range
                image = torch.clamp(image, 0, 1)

                # Convert to PIL Image
                pil_image = TF.to_pil_image(image)

                if self.enable_image_grid:
                    # Use key + step for Image Grid visualization
                    # WARNING: This triggers MLFlow's URL encoding bug (% characters)
                    # but enables the nice grid view in MLFlow UI
                    key = f"train_epoch_{epoch:03d}_img_{i:02d}"

                    mlflow_logger.experiment.log_image(
                        run_id=mlflow_logger.run_id,
                        image=pil_image,
                        key=key,
                        step=step,  # Required for Image Grid, but causes URL encoding bug
                    )
                else:
                    # Use artifact_file for clean URLs but no Image Grid
                    # This avoids the % encoding issues but images won't show in grid view
                    artifact_filename = f"{self.artifact_path}/epoch_{epoch:03d}/train_step_{step:06d}_img_{i:02d}.png"

                    mlflow_logger.experiment.log_image(
                        run_id=mlflow_logger.run_id,
                        image=pil_image,
                        artifact_file=artifact_filename,
                    )

                logged_count += 1

            except Exception as e:
                print(f"Warning: Failed to log image {i}: {e}")

        if logged_count > 0:
            mode = "Image Grid" if self.enable_image_grid else "artifacts"
            print(f"Logged {logged_count} images to MLFlow {mode} (epoch {epoch})")
            if self.enable_image_grid:
                print(
                    "  ⚠️  Note: Image Grid mode may cause URL encoding issues due to MLFlow bug"
                )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log training images if it's time to log."""
        if not self._should_log(trainer):
            return

        # Extract images from batch (ignore labels completely)
        if isinstance(batch, (tuple, list)):
            images = batch[0]  # Just take the first element (images)
        else:
            images = batch

        self._log_images_to_mlflow(images, trainer)
        self._last_logged_epoch = trainer.current_epoch
