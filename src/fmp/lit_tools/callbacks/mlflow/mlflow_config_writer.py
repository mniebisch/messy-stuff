from pathlib import Path
from typing import Optional

import lightning as L
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import MLFlowLogger
from typing_extensions import override

__all__ = ["MLFlowConfigWriter"]


class MLFlowConfigWriter(SaveConfigCallback):
    """
    Enhanced config writer for MLFlow that saves configuration as both file and artifact.

    This callback extends SaveConfigCallback to:
    1. Save config files to a local directory structure organized by run ID
    2. Log the config as an MLFlow artifact
    3. Set MLFlow tags for easy filtering and identification
    """

    def __init__(
        self,
        *args,
        logdir: str = "./config_logs",
        artifact_path: str = "configs",
        config_filename: str = "config.yaml",
        log_as_artifact: bool = True,
        set_config_tags: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logdir = Path(logdir).resolve()
        self.artifact_path = artifact_path
        self.config_filename = config_filename
        self.log_as_artifact = log_as_artifact
        self.set_config_tags = set_config_tags

    @override
    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        if self.already_saved:
            return

        # Only proceed during training or testing
        if stage not in ["fit", "test"]:
            return

        # Get MLFlow logger
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            # Fall back to standard behavior if no MLFlow logger
            super().setup(trainer, pl_module, stage)
            return

        run_id = mlflow_logger.run_id
        experiment_name = mlflow_logger.experiment.get_experiment(
            mlflow_logger.experiment_id
        ).name

        if self.save_to_log_dir and trainer.is_global_zero:
            # Create directory structure: logdir/experiment_name/run_id/
            log_dir = self.logdir / experiment_name / run_id
            log_dir.mkdir(parents=True, exist_ok=True)

            config_path = log_dir / self.config_filename

            if not self.overwrite and config_path.exists():
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. "
                    f"Aborting to avoid overwriting results of a previous run. "
                    f"You can delete the previous config file, set overwrite=True, "
                    f"or set save_config_callback=None to disable config saving."
                )

            # Save config to local file
            self.parser.save(
                self.config,
                str(config_path),
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )

            # Log config as MLFlow artifact
            if self.log_as_artifact:
                self._log_config_as_artifact(mlflow_logger, config_path)

            # Set helpful MLFlow tags
            if self.set_config_tags:
                self._set_config_tags(mlflow_logger, config_path)

        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True

        # Broadcast so that all ranks are in sync
        self.already_saved = trainer.strategy.broadcast(self.already_saved)

    def _get_mlflow_logger(self, trainer: L.Trainer) -> Optional[MLFlowLogger]:
        """Extract MLFlow logger from trainer."""
        if isinstance(trainer.logger, MLFlowLogger):
            return trainer.logger
        elif hasattr(trainer.logger, "__iter__"):
            # Handle multiple loggers
            for logger in trainer.logger:
                if isinstance(logger, MLFlowLogger):
                    return logger
        return None

    def _log_config_as_artifact(
        self, mlflow_logger: MLFlowLogger, config_path: Path
    ) -> None:
        """Log configuration file as MLFlow artifact."""
        try:
            mlflow_logger.experiment.log_artifact(
                run_id=mlflow_logger.run_id,
                local_path=str(config_path),
                artifact_path=self.artifact_path,
            )
        except Exception as e:
            # Log warning but don't fail training
            print(f"Warning: Failed to log config as artifact: {e}")

    def _set_config_tags(self, mlflow_logger: MLFlowLogger, config_path: Path) -> None:
        """Set useful MLFlow tags related to configuration."""
        try:
            # Set tags for easy filtering
            mlflow_logger.experiment.set_tag(
                mlflow_logger.run_id, "config.filename", self.config_filename
            )
            mlflow_logger.experiment.set_tag(
                mlflow_logger.run_id, "config.local_path", str(config_path)
            )
            mlflow_logger.experiment.set_tag(
                mlflow_logger.run_id, "config.artifact_path", self.artifact_path
            )

            # Optionally add a config hash for versioning
            if config_path.exists():
                config_hash = self._compute_config_hash(config_path)
                mlflow_logger.experiment.set_tag(
                    mlflow_logger.run_id, "config.hash", config_hash
                )
        except Exception as e:
            print(f"Warning: Failed to set config tags: {e}")

    def _compute_config_hash(self, config_path: Path) -> str:
        """Compute SHA256 hash of config file for versioning."""
        import hashlib

        content = config_path.read_text(encoding="utf-8")
        return hashlib.sha256(content.encode()).hexdigest()[
            :16
        ]  # Truncated for readability
