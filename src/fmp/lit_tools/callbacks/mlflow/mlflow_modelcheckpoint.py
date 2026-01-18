"""Enhanced ModelCheckpoint for MLFlow integration.

This module provides an MLFlow-aware checkpoint callback that organizes
checkpoints by experiment and run, with optional artifact logging.

Example:
    >>> from fmp.lit_tools.callbacks.mlflow import MLFlowModelCheckpoint
    >>>
    >>> checkpoint_callback = MLFlowModelCheckpoint(
    ...     dirpath="./checkpoints",
    ...     monitor="val_loss",
    ...     mode="min",
    ...     save_top_k=3,
    ...     log_best_as_artifact=True,
    ... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

__all__ = ["MLFlowModelCheckpoint"]

logger = logging.getLogger(__name__)


class MLFlowModelCheckpoint(ModelCheckpoint):
    """Enhanced ModelCheckpoint for MLFlow with experiment/run organization.

    This callback extends :class:`~lightning.pytorch.callbacks.ModelCheckpoint` to:

    1. Organize checkpoint directories by ``experiment_name/run_id``
    2. Log checkpoint paths as MLFlow tags for easy retrieval
    3. Optionally log checkpoints as MLFlow artifacts

    The directory structure created is::

        base_dirpath/
        └── experiment_name/
            └── run_id/
                ├── checkpoint-epoch=01-val_loss=0.50.ckpt
                └── checkpoint-epoch=02-val_loss=0.45.ckpt

    Artifact Logging Strategies:
        - ``log_best_as_artifact=True``: Only the single best checkpoint (minimal storage)
        - ``log_top_k_as_artifacts=True``: Same top-k as local (matches save_top_k setting)
        - ``log_improvement_history=True``: Every checkpoint that was ever the best

    Note:
        Only one artifact logging strategy can be enabled at a time.

    Attributes:
        base_dirpath: Base directory for checkpoint storage.
        artifact_path: Path within MLFlow artifacts for checkpoint storage.
    """

    def __init__(
        self,
        *args: Any,
        dirpath: str | None = None,
        log_checkpoint_tags: bool = True,
        log_best_as_artifact: bool = False,
        log_top_k_as_artifacts: bool = False,
        log_improvement_history: bool = False,
        artifact_path: str = "checkpoints",
        **kwargs: Any,
    ) -> None:
        """Initialize the MLFlow-aware checkpoint callback.

        Args:
            *args: Positional arguments passed to parent ModelCheckpoint.
            dirpath: Base directory for checkpoints. Will be organized by
                ``experiment_name/run_id`` when MLFlowLogger is present.
            log_checkpoint_tags: Whether to log checkpoint paths as MLFlow tags.
            log_best_as_artifact: Whether to log only the single best checkpoint
                as artifact (minimal storage).
            log_top_k_as_artifacts: Whether to log the same top-k checkpoints as
                saved locally (matches save_top_k setting).
            log_improvement_history: Whether to log every checkpoint that becomes
                a new best (useful for research/analysis).
            artifact_path: Path within MLFlow artifacts to store checkpoints.
            **kwargs: Keyword arguments passed to parent ModelCheckpoint.

        Raises:
            ValueError: If more than one artifact logging strategy is enabled.
        """
        # Store base_dirpath for later use in setup()
        if dirpath is not None:
            self.base_dirpath = Path(dirpath)
        else:
            self.base_dirpath = Path("./checkpoints")

        self.log_checkpoint_tags = log_checkpoint_tags
        self.log_best_as_artifact = log_best_as_artifact
        self.log_top_k_as_artifacts = log_top_k_as_artifacts
        self.log_improvement_history = log_improvement_history
        self.artifact_path = artifact_path

        # Validate conflicting options
        artifact_options = [
            log_best_as_artifact,
            log_top_k_as_artifacts,
            log_improvement_history,
        ]
        if sum(artifact_options) > 1:
            raise ValueError(
                "Only one artifact logging strategy can be enabled at a time: "
                "log_best_as_artifact, log_top_k_as_artifacts, or log_improvement_history"
            )

        # Don't pass dirpath to parent - we'll set it in setup() after getting MLFlow info
        super().__init__(*args, **kwargs)

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        """Set up checkpoint directory structure based on MLFlow run info.

        Creates a directory structure organized by experiment name and run ID
        when an MLFlowLogger is present. Falls back to a default directory
        structure otherwise.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The LightningModule being trained.
            stage: Current stage ('fit', 'validate', 'test', 'predict').
        """
        mlflow_logger = self._get_mlflow_logger(trainer)

        if mlflow_logger is not None:
            experiment_id = mlflow_logger.experiment_id
            run_id = mlflow_logger.run_id

            if experiment_id is None or run_id is None:
                logger.warning(
                    "MLFlow experiment_id or run_id is None, using default checkpoint path"
                )
                self.dirpath = str(self.base_dirpath / "default")
            else:
                # Get experiment name
                experiment = mlflow_logger.experiment.get_experiment(experiment_id)
                experiment_name = experiment.name if experiment else "unknown"

                # Set checkpoint directory: base_dirpath/experiment_name/run_id/
                self.dirpath = str(self.base_dirpath / experiment_name / run_id)

                # Log checkpoint directory as MLFlow tag
                if self.log_checkpoint_tags:
                    try:
                        mlflow_logger.experiment.set_tag(
                            run_id, "checkpoint.dirpath", self.dirpath
                        )
                    except Exception as e:
                        logger.warning(f"Failed to set checkpoint dirpath tag: {e}")
        else:
            # Fallback if no MLFlow logger
            self.dirpath = str(self.base_dirpath / "default")

        # Ensure directory exists
        Path(self.dirpath).mkdir(parents=True, exist_ok=True)

        # Log clear path information for user (only on rank zero)
        if trainer.is_global_zero and mlflow_logger is not None:
            experiment_id = mlflow_logger.experiment_id
            run_id = mlflow_logger.run_id
            if experiment_id and run_id:
                experiment = mlflow_logger.experiment.get_experiment(experiment_id)
                experiment_name = experiment.name if experiment else "unknown"
                mlflow_artifacts_path = (
                    f"mlruns/{experiment_id}/{run_id}/artifacts/{self.artifact_path}"
                )

                strategy = "none"
                if self.log_best_as_artifact:
                    strategy = "best_only"
                elif self.log_improvement_history:
                    strategy = "improvement_history"
                elif self.log_top_k_as_artifacts:
                    strategy = "top_k"

                logger.info("📁 MLFlowModelCheckpoint Setup:")
                logger.info(f"   🏠 Local checkpoints: {self.dirpath}")
                logger.info(f"   ☁️  MLFlow artifacts: {mlflow_artifacts_path}")
                logger.info(f"   🎯 Strategy: {strategy}")

        # Call parent setup
        super().setup(trainer, pl_module, stage)

    def _update_best_and_save(
        self, current: Any, trainer: L.Trainer, monitor_candidates: dict[str, Any]
    ) -> None:
        """Override to log checkpoint info to MLFlow based on selected strategy.

        Args:
            current: Current metric value.
            trainer: The PyTorch Lightning trainer instance.
            monitor_candidates: Dictionary of metric candidates.
        """
        # Store the old best path to detect changes
        old_best_path = getattr(self, "best_model_path", None)

        # Call parent method
        super()._update_best_and_save(current, trainer, monitor_candidates)

        # Check if the best model path actually changed
        new_best_path = getattr(self, "best_model_path", None)
        best_model_changed = old_best_path != new_best_path

        # Log additional MLFlow tags and artifacts
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is not None and trainer.is_global_zero:
            try:
                # Log tags if enabled
                if self.log_checkpoint_tags and new_best_path:
                    self._log_checkpoint_tags(mlflow_logger, new_best_path)

                # Handle artifact logging based on strategy
                if self.log_best_as_artifact and best_model_changed and new_best_path:
                    self._log_best_checkpoint_artifact(mlflow_logger, new_best_path)
                elif (
                    self.log_improvement_history
                    and best_model_changed
                    and new_best_path
                ):
                    self._log_improvement_history_artifact(mlflow_logger, new_best_path)
                # Note: log_top_k_as_artifacts is handled in on_train_end

            except Exception as e:
                logger.warning(f"Failed to log checkpoint info to MLFlow: {e}")

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log top-k artifacts at the end of training.

        This is called once at the end of training to avoid duplicate artifact
        uploads during the training loop.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The LightningModule being trained.
        """
        if self.log_top_k_as_artifacts:
            mlflow_logger = self._get_mlflow_logger(trainer)
            if mlflow_logger is not None and trainer.is_global_zero:
                try:
                    self._log_top_k_artifacts(mlflow_logger)
                except Exception as e:
                    logger.warning(f"Failed to log top-k artifacts to MLFlow: {e}")

    def _log_checkpoint_tags(self, mlflow_logger: MLFlowLogger, best_path: str) -> None:
        """Log checkpoint information as MLFlow tags.

        Args:
            mlflow_logger: The MLFlow logger instance.
            best_path: Path to the best checkpoint file.
        """
        run_id = mlflow_logger.run_id
        if run_id is None:
            logger.warning("Cannot log checkpoint tags: run_id is None")
            return

        if best_path:
            mlflow_logger.experiment.set_tag(
                run_id,
                "checkpoint.best_model_path",
                str(best_path),
            )

        if hasattr(self, "best_model_score") and self.best_model_score is not None:
            score_value = (
                self.best_model_score.cpu().item()
                if hasattr(self.best_model_score, "cpu")
                else self.best_model_score
            )
            mlflow_logger.experiment.set_tag(
                run_id,
                f"checkpoint.best_{self.monitor}",
                str(score_value),
            )

    def _log_best_checkpoint_artifact(
        self, mlflow_logger: MLFlowLogger, best_path: str
    ) -> None:
        """Log only the single best checkpoint as artifact.

        Args:
            mlflow_logger: The MLFlow logger instance.
            best_path: Path to the best checkpoint file.
        """
        run_id = mlflow_logger.run_id
        if run_id is None:
            logger.warning("Cannot log artifact: run_id is None")
            return

        if best_path and Path(best_path).exists():
            mlflow_logger.experiment.log_artifact(
                run_id=run_id,
                local_path=str(best_path),
                artifact_path=self.artifact_path,
            )
            logger.info(
                f"📦 Logged best checkpoint as artifact: {Path(best_path).name}"
            )

    def _log_improvement_history_artifact(
        self, mlflow_logger: MLFlowLogger, best_path: str
    ) -> None:
        """Log every checkpoint that becomes a new best.

        This logs each improvement as it happens, creating a history of
        all checkpoints that were ever the best at some point.

        Args:
            mlflow_logger: The MLFlow logger instance.
            best_path: Path to the newly best checkpoint file.
        """
        run_id = mlflow_logger.run_id
        if run_id is None:
            logger.warning("Cannot log artifact: run_id is None")
            return

        if best_path and Path(best_path).exists():
            mlflow_logger.experiment.log_artifact(
                run_id=run_id,
                local_path=str(best_path),
                artifact_path=self.artifact_path,
            )
            logger.info(
                f"📈 Logged improved checkpoint as artifact: {Path(best_path).name}"
            )

    def _log_top_k_artifacts(self, mlflow_logger: MLFlowLogger) -> None:
        """Log the same top-k checkpoints that are saved locally.

        Called once at end of training to upload all top-k checkpoints
        to MLFlow artifacts.

        Args:
            mlflow_logger: The MLFlow logger instance.
        """
        run_id = mlflow_logger.run_id
        if run_id is None:
            logger.warning("Cannot log top-k artifacts: run_id is None")
            return

        if not hasattr(self, "best_k_models") or not self.best_k_models:
            logger.warning("No best_k_models found, skipping top-k artifact logging")
            return

        # Get all current top-k checkpoint paths
        current_top_k = list(self.best_k_models.keys())

        logger.info(
            f"📊 Logging {len(current_top_k)} top-k checkpoints as artifacts "
            f"(save_top_k={self.save_top_k})"
        )

        if current_top_k:
            logger.info(f"   🏠 From local directory: {Path(current_top_k[0]).parent}")
            logger.info(
                f"   ☁️  To MLFlow artifacts: mlruns/{mlflow_logger.experiment_id}/"
                f"{run_id}/artifacts/{self.artifact_path}"
            )

        for checkpoint_path in current_top_k:
            if checkpoint_path and Path(checkpoint_path).exists():
                mlflow_logger.experiment.log_artifact(
                    run_id=run_id,
                    local_path=str(checkpoint_path),
                    artifact_path=self.artifact_path,
                )
                logger.info(
                    f"🎯 Logged top-k checkpoint as artifact: {Path(checkpoint_path).name}"
                )
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")

    def _get_mlflow_logger(self, trainer: L.Trainer) -> MLFlowLogger | None:
        """Extract MLFlow logger from trainer.

        Args:
            trainer: The PyTorch Lightning trainer instance.

        Returns:
            The MLFlowLogger if found, None otherwise.
        """
        if trainer.logger is None:
            return None

        if isinstance(trainer.logger, MLFlowLogger):
            return trainer.logger

        # Handle multiple loggers
        if hasattr(trainer.logger, "__iter__"):
            for log_instance in trainer.logger:
                if isinstance(log_instance, MLFlowLogger):
                    return log_instance

        return None
