from pathlib import Path
from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

__all__ = ["MLFlowModelCheckpoint"]


class MLFlowModelCheckpoint(ModelCheckpoint):
    """
    Enhanced ModelCheckpoint for MLFlow that organizes checkpoints by experiment and run.

    This callback extends ModelCheckpoint to:
    1. Organize checkpoint directories by experiment_name/run_id
    2. Log checkpoint paths as MLFlow tags
    3. Optionally log the best checkpoint as MLFlow artifact

    The log_best_as_artifact option logs only the single best checkpoint to MLFlow artifacts,
    not all top-k checkpoints. This provides a good balance between functionality and storage efficiency.
    """

    def __init__(
        self,
        *args,
        base_dirpath: Optional[str] = None,
        dirpath: Optional[
            str
        ] = None,  # For compatibility with standard ModelCheckpoint
        log_checkpoint_tags: bool = True,
        log_best_as_artifact: bool = False,
        log_top_k_as_artifacts: bool = False,
        log_improvement_history: bool = False,
        artifact_path: str = "checkpoints",
        **kwargs,
    ):
        """
        Args:
            base_dirpath: Base directory for checkpoints (will be organized by experiment/run)
            dirpath: Alternative to base_dirpath for compatibility with ModelCheckpoint
            log_checkpoint_tags: Whether to log checkpoint paths as MLFlow tags
            log_best_as_artifact: Whether to log only the single best checkpoint as artifact
            log_top_k_as_artifacts: Whether to log the same top-k checkpoints as saved locally
            log_improvement_history: Whether to log every checkpoint that becomes a new best
            artifact_path: Path within MLFlow artifacts to store checkpoints
            **kwargs: Additional arguments passed to ModelCheckpoint

        Artifact Logging Strategies:
            - log_best_as_artifact=True: Only the single best checkpoint (minimal storage)
            - log_top_k_as_artifacts=True: Same top-k as local (matches save_top_k setting)
            - log_improvement_history=True: Every checkpoint that was ever the best (research/analysis)

        Note: Only one of the artifact logging options should be enabled at a time.
        """
        # Handle dirpath vs base_dirpath for compatibility
        if base_dirpath is not None and dirpath is not None:
            raise ValueError(
                "Cannot specify both 'base_dirpath' and 'dirpath'. Use 'base_dirpath' for MLFlow organization."
            )

        if base_dirpath is not None:
            self.base_dirpath = Path(base_dirpath)
        elif dirpath is not None:
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

        # dirpath will be set in setup() based on MLFlow run info
        super().__init__(*args, dirpath=None, **kwargs)

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        """Set up checkpoint directory structure based on MLFlow run info."""
        mlflow_logger = self._get_mlflow_logger(trainer)

        if mlflow_logger is not None:
            # Get experiment name and run ID
            experiment_name = mlflow_logger.experiment.get_experiment(
                mlflow_logger.experiment_id
            ).name
            run_id = mlflow_logger.run_id

            # Set checkpoint directory: base_dirpath/experiment_name/run_id/
            self.dirpath = str(self.base_dirpath / experiment_name / run_id)

            # Log checkpoint directory as MLFlow tag
            if self.log_checkpoint_tags:
                try:
                    mlflow_logger.experiment.set_tag(
                        run_id, "checkpoint.dirpath", self.dirpath
                    )
                except Exception as e:
                    print(f"Warning: Failed to set checkpoint dirpath tag: {e}")
        else:
            # Fallback if no MLFlow logger
            self.dirpath = str(self.base_dirpath / "default")

        # Ensure directory exists
        Path(self.dirpath).mkdir(parents=True, exist_ok=True)

        # Print clear path information for user
        if mlflow_logger is not None:
            experiment_name = mlflow_logger.experiment.get_experiment(
                mlflow_logger.experiment_id
            ).name
            run_id = mlflow_logger.run_id
            mlflow_artifacts_path = f"mlruns/{mlflow_logger.experiment_id}/{run_id}/artifacts/{self.artifact_path}"

            print("📁 MLFlowModelCheckpoint Setup:")
            print(f"   🏠 Local checkpoints: {self.dirpath}")
            print(f"   ☁️  MLFlow artifacts: {mlflow_artifacts_path}")
            print(
                f"   🎯 Strategy: {'best_only' if self.log_best_as_artifact else 'improvement_history' if self.log_improvement_history else 'top_k' if self.log_top_k_as_artifacts else 'none'}"
            )

        # Call parent setup
        super().setup(trainer, pl_module, stage)

    def _update_best_and_save(self, current, trainer, monitor_candidates):
        """Override to log checkpoint info to MLFlow based on selected strategy."""
        # Store the old best path to detect changes
        old_best_path = getattr(self, "best_model_path", None)

        # Call parent method
        super()._update_best_and_save(current, trainer, monitor_candidates)

        # Check if the best model path actually changed
        new_best_path = getattr(self, "best_model_path", None)
        best_model_changed = old_best_path != new_best_path

        # Log additional MLFlow tags and artifacts
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger and trainer.is_global_zero:
            try:
                # Log tags if enabled
                if self.log_checkpoint_tags:
                    self._log_checkpoint_tags(mlflow_logger, new_best_path)

                # Handle artifact logging based on strategy
                if self.log_best_as_artifact and best_model_changed:
                    self._log_best_checkpoint_artifact(mlflow_logger, new_best_path)
                elif self.log_improvement_history and best_model_changed:
                    self._log_improvement_history_artifact(mlflow_logger, new_best_path)
                # Note: log_top_k_as_artifacts is handled in on_train_end to avoid duplicates

            except Exception as e:
                print(f"Warning: Failed to log checkpoint info to MLFlow: {e}")

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log top-k artifacts at the end of training to avoid duplicates."""
        if self.log_top_k_as_artifacts:
            mlflow_logger = self._get_mlflow_logger(trainer)
            if mlflow_logger and trainer.is_global_zero:
                try:
                    self._log_top_k_artifacts(mlflow_logger)
                except Exception as e:
                    print(f"Warning: Failed to log top-k artifacts to MLFlow: {e}")

    def _log_checkpoint_tags(self, mlflow_logger: MLFlowLogger, best_path: str):
        """Log checkpoint information as MLFlow tags."""
        if best_path:
            mlflow_logger.experiment.set_tag(
                mlflow_logger.run_id,
                "checkpoint.best_model_path",
                str(best_path),
            )

        if hasattr(self, "best_model_score") and self.best_model_score is not None:
            mlflow_logger.experiment.set_tag(
                mlflow_logger.run_id,
                f"checkpoint.best_{self.monitor}",
                str(
                    self.best_model_score.cpu().item()
                    if hasattr(self.best_model_score, "cpu")
                    else self.best_model_score
                ),
            )

    def _log_best_checkpoint_artifact(
        self, mlflow_logger: MLFlowLogger, best_path: str
    ):
        """Log only the single best checkpoint as artifact."""
        if best_path and Path(best_path).exists():
            mlflow_logger.experiment.log_artifact(
                run_id=mlflow_logger.run_id,
                local_path=str(best_path),
                artifact_path=self.artifact_path,
            )
            print(f"📦 Logged best checkpoint as artifact: {Path(best_path).name}")

    def _log_improvement_history_artifact(
        self, mlflow_logger: MLFlowLogger, best_path: str
    ):
        """Log every checkpoint that becomes a new best (previous behavior)."""
        if best_path and Path(best_path).exists():
            mlflow_logger.experiment.log_artifact(
                run_id=mlflow_logger.run_id,
                local_path=str(best_path),
                artifact_path=self.artifact_path,
            )
            print(f"📈 Logged improved checkpoint as artifact: {Path(best_path).name}")

    def _log_top_k_artifacts(self, mlflow_logger: MLFlowLogger):
        """Log the same top-k checkpoints that are saved locally (called once at end of training)."""
        if not hasattr(self, "best_k_models") or not self.best_k_models:
            print("⚠️  No best_k_models found, skipping top-k artifact logging")
            return

        # Get all current top-k checkpoint paths
        current_top_k = [
            checkpoint_path for checkpoint_path, _ in self.best_k_models.items()
        ]

        print(
            f"📊 Logging {len(current_top_k)} top-k checkpoints as artifacts (save_top_k={self.save_top_k})"
        )
        print(
            f"   🏠 From local directory: {Path(current_top_k[0]).parent if current_top_k else 'N/A'}"
        )
        print(
            f"   ☁️  To MLFlow artifacts: mlruns/{mlflow_logger.experiment_id}/{mlflow_logger.run_id}/artifacts/{self.artifact_path}"
        )

        for checkpoint_path in current_top_k:
            if checkpoint_path and Path(checkpoint_path).exists():
                mlflow_logger.experiment.log_artifact(
                    run_id=mlflow_logger.run_id,
                    local_path=str(checkpoint_path),
                    artifact_path=self.artifact_path,
                )
                print(
                    f"🎯 Logged top-k checkpoint as artifact: {Path(checkpoint_path).name}"
                )
            else:
                print(f"⚠️  Checkpoint not found: {checkpoint_path}")

    def _get_mlflow_logger(self, trainer: L.Trainer) -> Optional[MLFlowLogger]:
        """Extract MLFlow logger from trainer."""
        if isinstance(trainer.logger, MLFlowLogger):
            return trainer.logger
        elif trainer.logger and hasattr(trainer.logger, "__iter__"):
            # Handle multiple loggers
            for logger in trainer.logger:
                if isinstance(logger, MLFlowLogger):
                    return logger
        return None
