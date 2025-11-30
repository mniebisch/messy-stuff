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
    ):
        """
        Args:
            log_every_n_epochs: Log images every N epochs
            max_images_per_batch: Maximum number of images to log per batch
            denormalize: Whether to denormalize images for logging
            normalize_mean: Mean values used for normalization (for denormalization)
            normalize_std: Std values used for normalization (for denormalization)
            artifact_path: Base path within MLFlow artifacts for images
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
        """Log images to MLFlow using log_image for both artifacts and metrics."""
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        # Take first N images from batch
        n_images = min(self.max_images_per_batch, images.shape[0])
        sample_images = images[:n_images]

        # Process and log each image individually
        for i, image in enumerate(sample_images):
            try:
                # Denormalize if needed
                if self.denormalize:
                    image = self._denormalize_image(image)

                # Ensure image is in [0, 1] range
                image = torch.clamp(image, 0, 1)

                # Convert to PIL Image
                pil_image = TF.to_pil_image(image)

                # Create a key for the image (this enables metrics/charts)
                epoch = trainer.current_epoch
                step = trainer.global_step
                image_key = f"training_images/epoch_{epoch}_img_{i}"

                # Log image using MLFlow's log_image with step (enables charts)
                mlflow_logger.experiment.log_image(
                    run_id=mlflow_logger.run_id,
                    image=pil_image,
                    key=image_key,
                    step=step,
                )

            except Exception as e:
                print(f"Warning: Failed to log image {i} to MLFlow: {e}")

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
