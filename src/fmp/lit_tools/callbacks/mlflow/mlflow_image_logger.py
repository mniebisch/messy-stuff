"""
Simple MLFlow image logging callback for PyTorch Lightning.

This module provides a simple callback for logging training images to MLFlow.
"""

import math
from typing import Any, List, Optional

import lightning as L
import torch
import torchvision.transforms.functional as TF
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger
from PIL import Image, ImageDraw, ImageFont

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
        create_composite_grid: bool = True,
        grid_cols: int = 4,
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
            create_composite_grid: Create a single composite image showing all images in a grid
            grid_cols: Number of columns in the composite grid
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
        self.create_composite_grid = create_composite_grid
        self.grid_cols = grid_cols

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

    def _create_composite_grid(
        self, pil_images: list, epoch: int, step: int
    ) -> Image.Image:
        """Create a composite image showing multiple images in a grid layout."""
        if not pil_images:
            return None

        n_images = len(pil_images)
        cols = min(self.grid_cols, n_images)
        rows = math.ceil(n_images / cols)

        # Get image dimensions (assume all images are same size)
        img_width, img_height = pil_images[0].size

        # Calculate grid dimensions with padding
        padding = 10
        grid_width = cols * img_width + (cols + 1) * padding
        grid_height = (
            rows * img_height + (rows + 1) * padding + 50
        )  # Extra space for title

        # Create composite image with white background
        composite = Image.new("RGB", (grid_width, grid_height), color="white")
        draw = ImageDraw.Draw(composite)

        # Add title
        title = f"Training Images - Epoch {epoch:03d}, Step {step:06d}"
        try:
            # Try to use a system font, fall back to default if not available
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        # Center the title
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (grid_width - title_width) // 2
        draw.text((title_x, 10), title, fill="black", font=font)

        # Place images in grid
        y_offset = 50  # After title
        for i, img in enumerate(pil_images):
            row = i // cols
            col = i % cols

            x = padding + col * (img_width + padding)
            y = y_offset + padding + row * (img_height + padding)

            composite.paste(img, (x, y))

        return composite

    def _log_images_to_mlflow(self, images: torch.Tensor, trainer: L.Trainer):
        """Log images to MLFlow using multiple strategies for best coverage."""
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        # Take first N images from batch
        n_images = min(self.max_images_per_batch, images.shape[0])
        sample_images = images[:n_images]

        epoch = trainer.current_epoch
        step = trainer.global_step
        pil_images = []
        logged_count = 0

        # Process all images first
        for i, image in enumerate(sample_images):
            try:
                # Denormalize if needed
                if self.denormalize:
                    image = self._denormalize_image(image)

                # Ensure image is in [0, 1] range
                image = torch.clamp(image, 0, 1)

                # Convert to PIL Image
                pil_image = TF.to_pil_image(image)
                pil_images.append(pil_image)

                # Strategy 1: Always log as individual artifacts (reliable)
                artifact_filename = f"{self.artifact_path}/epoch_{epoch:03d}/individual/train_step_{step:06d}_img_{i:02d}.png"
                mlflow_logger.experiment.log_image(
                    run_id=mlflow_logger.run_id,
                    image=pil_image,
                    artifact_file=artifact_filename,
                )

                # Strategy 2: Optionally log to Image Grid (buggy but nice when it works)
                if self.enable_image_grid:
                    key = f"train_epoch_{epoch:03d}_img_{i:02d}"
                    try:
                        mlflow_logger.experiment.log_image(
                            run_id=mlflow_logger.run_id,
                            image=pil_image,
                            key=key,
                            step=step,
                        )
                    except Exception as e:
                        print(f"Warning: Image Grid logging failed for image {i}: {e}")

                logged_count += 1

            except Exception as e:
                print(f"Warning: Failed to process image {i}: {e}")

        # Strategy 3: Create and log composite grid image (reliable grid view)
        if self.create_composite_grid and pil_images:
            try:
                composite_image = self._create_composite_grid(pil_images, epoch, step)
                if composite_image:
                    composite_filename = f"{self.artifact_path}/epoch_{epoch:03d}/composite_grid_step_{step:06d}.png"
                    mlflow_logger.experiment.log_image(
                        run_id=mlflow_logger.run_id,
                        image=composite_image,
                        artifact_file=composite_filename,
                    )
            except Exception as e:
                print(f"Warning: Failed to create composite grid: {e}")

        if logged_count > 0:
            strategies = []
            strategies.append("individual artifacts")
            if self.enable_image_grid:
                strategies.append("Image Grid")
            if self.create_composite_grid:
                strategies.append("composite grid")

            strategy_str = " + ".join(strategies)
            print(
                f"Logged {logged_count} images to MLFlow via {strategy_str} (epoch {epoch})"
            )

            if self.enable_image_grid:
                print(
                    "  ⚠️  Note: Image Grid mode may have URL encoding issues due to MLFlow bug"
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
