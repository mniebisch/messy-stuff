"""
Unified MLFlow configuration callback for PyTorch Lightning.

This module provides a single, comprehensive callback that handles all configuration
logging needs for MLFlow, replacing the previous separate callbacks.

Features:
    - Saves config to local filesystem (organized by MLFlow run ID)
    - Logs config as MLFlow artifact for experiment reproducibility
    - Sets MLFlow tags for filtering and metadata queries
    - Supports both CLI-generated and custom configs
    - Optional config hashing for duplicate detection

Example:
    >>> from lightning.pytorch.cli import LightningCLI
    >>> from fmp.lit_tools.callbacks.mlflow import MLFlowConfigCallback
    >>>
    >>> LightningCLI(
    ...     save_config_callback=MLFlowConfigCallback,
    ...     save_config_kwargs={
    ...         "logdir": "./config_logs",
    ...         "artifact_path": "configs",
    ...         "log_as_artifact": True,
    ...     },
    ... )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import lightning as L
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import MLFlowLogger
from typing_extensions import override

__all__ = ["MLFlowConfigCallback"]

logger = logging.getLogger(__name__)


class MLFlowConfigCallback(SaveConfigCallback):
    """Unified configuration callback for MLFlow logging.

    This callback combines the functionality of config saving and MLFlow logging,
    providing a comprehensive solution for experiment configuration management.

    Features:
        1. Saves config to local filesystem (organized by run ID)
        2. Logs config as MLFlow artifact
        3. Sets MLFlow tags for filtering and metadata
        4. Supports both CLI-generated and custom configs
        5. Optional config hashing for duplicate detection (disabled by default)

    Note:
        This callback is designed to be used as the ``save_config_callback`` parameter
        of :class:`~lightning.pytorch.cli.LightningCLI`. It extends
        :class:`~lightning.pytorch.cli.SaveConfigCallback` to add MLFlow-specific
        functionality.

    Attributes:
        logdir: Resolved path to local config directory.
        artifact_path: Path within MLFlow artifacts for config storage.
        config_filename: Name of the saved config file.
    """

    def __init__(
        self,
        *args: Any,
        # Local filesystem options
        logdir: str = "./config_logs",
        # MLFlow artifact options
        artifact_path: str = "artifacts",
        config_filename: str = "config.yaml",
        log_as_artifact: bool = True,
        # MLFlow tagging options
        tag_prefix: str = "config",
        log_config_hash: bool = False,
        log_summary_tags: bool = True,
        # Additional config options
        resolved_config: dict[str, Any] | None = None,
        log_resolved_config: bool = True,
        pretty_json: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the MLFlow configuration callback.

        Args:
            *args: Positional arguments passed to parent SaveConfigCallback.
            logdir: Local directory to save configs (organized by run ID).
            artifact_path: Path within MLFlow artifacts to store config.
            config_filename: Name of the main config file.
            log_as_artifact: Whether to log config as MLFlow artifact.
            tag_prefix: Prefix for MLFlow tags (e.g., "config.model").
            log_config_hash: Whether to compute and log config hash.
                Disabled by default since MLFlow run ID provides uniqueness.
            log_summary_tags: Whether to log summary information as tags.
            resolved_config: Additional resolved config dict to log (from CLI).
            log_resolved_config: Whether to log the resolved config as JSON.
            pretty_json: Whether to format JSON with indentation.
            **kwargs: Keyword arguments passed to parent SaveConfigCallback.
        """
        super().__init__(*args, **kwargs)

        # Local filesystem settings
        self.logdir = Path(logdir).resolve()

        # MLFlow settings
        self.artifact_path = artifact_path
        self.config_filename = config_filename
        self.log_as_artifact = log_as_artifact

        # Tagging settings
        self.tag_prefix = tag_prefix
        self.log_config_hash = log_config_hash
        self.log_summary_tags = log_summary_tags

        # Additional config settings
        self.resolved_config = resolved_config
        self.log_resolved_config = log_resolved_config
        self.pretty_json = pretty_json

        # Internal state
        self._mlflow_logged = False

    @override
    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        """Setup callback - handles both local saving and MLFlow logging."""
        if self.already_saved:
            return

        # Only proceed during fit or test
        if stage not in ["fit", "test"]:
            return

        # Handle local config saving (from parent class)
        self._setup_local_saving(trainer, pl_module, stage)

        # Handle MLFlow logging
        if not self._mlflow_logged:
            self._setup_mlflow_logging(trainer)
            self._mlflow_logged = True

    def _setup_local_saving(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        """Handle local filesystem config saving with run ID organization.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The LightningModule being trained.
            stage: Current stage ('fit', 'validate', 'test', 'predict').
        """
        if not self.save_to_log_dir:
            return

        # Get MLFlow logger and run ID
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            # Fall back to original behavior if no MLFlow logger
            super().setup(trainer, pl_module, stage)
            return

        run_id = mlflow_logger.run_id
        if run_id is None:
            logger.warning(
                "MLFlow run_id is None, falling back to default SaveConfigCallback behavior"
            )
            super().setup(trainer, pl_module, stage)
            return

        log_dir = str(self.logdir / run_id)
        config_path = os.path.join(log_dir, self.config_filename)
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            # Check if file exists
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. "
                    f"Aborting to avoid overwriting results of a previous run. "
                    f"You can delete the previous config file, set save_config_callback=None, "
                    f'or set save_config_kwargs={{"overwrite": True}} to overwrite.'
                )

        if trainer.is_global_zero:
            # Create directory and save config
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )

        # Call parent's save_config method
        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True

        # Broadcast state
        self.already_saved = trainer.strategy.broadcast(self.already_saved)

    def _setup_mlflow_logging(self, trainer: L.Trainer) -> None:
        """Handle MLFlow logging of configuration.

        Logs the config as artifact, resolved config if provided, and summary tags.

        Args:
            trainer: The PyTorch Lightning trainer instance.
        """
        if not trainer.is_global_zero:
            return

        mlflow_logger = self._get_mlflow_logger(trainer)
        if not mlflow_logger:
            return

        # Log main config as artifact
        if self.log_as_artifact and hasattr(self, "config"):
            self._log_config_as_artifact(
                mlflow_logger, self.config, self.config_filename
            )

        # Log resolved config if provided
        if self.log_resolved_config and self.resolved_config:
            resolved_filename = (
                f"resolved_{self.config_filename.replace('.yaml', '.json')}"
            )
            self._log_resolved_config_as_artifact(mlflow_logger, resolved_filename)

        # Log summary tags
        if self.log_summary_tags:
            self._log_summary_tags(mlflow_logger)

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

        # Check if logger is iterable (e.g., list of loggers)
        if hasattr(trainer.logger, "__iter__"):
            for log_instance in trainer.logger:
                if isinstance(log_instance, MLFlowLogger):
                    return log_instance

        return None

    def _log_config_as_artifact(
        self, mlflow_logger: MLFlowLogger, config: Any, filename: str
    ) -> None:
        """Log configuration as MLFlow artifact.

        Args:
            mlflow_logger: The MLFlow logger instance.
            config: The configuration to log.
            filename: Name for the artifact file.

        Raises:
            RuntimeError: If artifact logging fails and error handling is not possible.
        """
        temp_dir = tempfile.TemporaryDirectory()
        try:
            config_path = Path(temp_dir.name) / filename

            # Save config to temporary file
            if hasattr(self, "parser") and self.parser is not None:
                # Use parser to save in original format
                self.parser.save(config, str(config_path), skip_none=False)
            else:
                # Fallback to JSON
                config_path = config_path.with_suffix(".json")
                config_path.write_text(
                    json.dumps(
                        config, indent=2 if self.pretty_json else None, default=str
                    )
                )

            # Log as artifact
            run_id = mlflow_logger.run_id
            if run_id is None:
                logger.warning("Cannot log artifact: MLFlow run_id is None")
                return

            mlflow_logger.experiment.log_artifact(
                run_id=run_id,
                local_path=str(config_path),
                artifact_path=self.artifact_path,
            )

            # Log hash if enabled
            if self.log_config_hash:
                config_hash = self._compute_config_hash(config)
                mlflow_logger.experiment.set_tag(
                    run_id, f"{self.tag_prefix}.hash", config_hash
                )

        except Exception as e:
            logger.error(f"Failed to log config as artifact: {e}")
            raise
        finally:
            temp_dir.cleanup()

    def _log_resolved_config_as_artifact(
        self, mlflow_logger: MLFlowLogger, filename: str
    ) -> None:
        """Log resolved configuration as JSON artifact.

        Args:
            mlflow_logger: The MLFlow logger instance.
            filename: Name for the artifact file.
        """
        if not self.resolved_config:
            return

        run_id = mlflow_logger.run_id
        if run_id is None:
            logger.warning("Cannot log resolved config: MLFlow run_id is None")
            return

        temp_dir = tempfile.TemporaryDirectory()
        try:
            config_path = Path(temp_dir.name) / filename

            # Create JSON representation
            json_str = json.dumps(
                self.resolved_config,
                indent=2 if self.pretty_json else None,
                sort_keys=True,
                default=str,
            )
            config_path.write_text(json_str, encoding="utf-8")

            # Log as artifact
            mlflow_logger.experiment.log_artifact(
                run_id=run_id,
                local_path=str(config_path),
                artifact_path=self.artifact_path,
            )

            # Log resolved config hash
            if self.log_config_hash:
                config_hash = hashlib.sha256(json_str.encode()).hexdigest()
                mlflow_logger.experiment.set_tag(
                    run_id, f"{self.tag_prefix}.resolved_hash", config_hash
                )

        except Exception as e:
            logger.error(f"Failed to log resolved config as artifact: {e}")
        finally:
            temp_dir.cleanup()

    def _log_summary_tags(self, mlflow_logger: MLFlowLogger) -> None:
        """Log summary information as MLFlow tags.

        Extracts key information from config and logs as searchable tags.

        Args:
            mlflow_logger: The MLFlow logger instance.
        """
        run_id = mlflow_logger.run_id
        if run_id is None:
            logger.warning("Cannot log summary tags: MLFlow run_id is None")
            return

        tags: dict[str, str] = {}

        # Log from main config if available
        if hasattr(self, "config") and self.config:
            config_summary = self._extract_config_summary(self.config)
            for key, value in config_summary.items():
                tags[f"{self.tag_prefix}.{key}"] = str(value)

        # Log from resolved config if available
        if self.resolved_config:
            resolved_summary = self._extract_resolved_config_summary(
                self.resolved_config
            )
            for key, value in resolved_summary.items():
                tags[f"{self.tag_prefix}.resolved.{key}"] = str(value)

        # Set all tags
        for tag_key, tag_value in tags.items():
            try:
                mlflow_logger.experiment.set_tag(run_id, tag_key, tag_value)
            except Exception as e:
                logger.warning(f"Failed to set tag {tag_key}: {e}")

    def _compute_config_hash(self, config: Any) -> str:
        """Compute SHA-256 hash of configuration.

        Args:
            config: The configuration object to hash.

        Returns:
            Hexadecimal hash string of the configuration.
        """
        try:
            # Convert to JSON string for consistent hashing
            json_str = json.dumps(config, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()
        except Exception:
            # Fallback to string representation
            return hashlib.sha256(str(config).encode()).hexdigest()

    def _extract_config_summary(self, config: Any) -> dict[str, Any]:
        """Extract summary information from main config.

        Args:
            config: The configuration object (can be dict or namespace).

        Returns:
            Dictionary with extracted summary key-value pairs.
        """
        summary: dict[str, Any] = {}

        # Try to extract common config elements
        if hasattr(config, "__dict__"):
            config_dict = config.__dict__
        elif isinstance(config, dict):
            config_dict = config
        else:
            return summary

        # Extract key information
        for key in ["model", "data", "trainer", "optimizer"]:
            if key in config_dict:
                value = config_dict[key]
                if hasattr(value, "class_path"):
                    summary[key] = value.class_path
                elif hasattr(value, "__class__"):
                    summary[key] = value.__class__.__name__
                else:
                    summary[key] = str(type(value).__name__)

        return summary

    def _extract_resolved_config_summary(
        self, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract summary information from resolved config.

        Args:
            config: The resolved configuration dictionary.

        Returns:
            Dictionary with extracted summary key-value pairs.
        """
        summary: dict[str, Any] = {}

        # Extract key paths and parameters
        for section in ["model", "data", "trainer", "optimizer", "lr_scheduler"]:
            if section in config:
                section_config = config[section]
                if isinstance(section_config, dict):
                    if "class_path" in section_config:
                        summary[f"{section}_class"] = section_config["class_path"]
                    if "init_args" in section_config:
                        # Extract a few key parameters
                        init_args = section_config["init_args"]
                        if isinstance(init_args, dict):
                            for key in [
                                "lr",
                                "batch_size",
                                "max_epochs",
                                "num_classes",
                            ]:
                                if key in init_args:
                                    summary[f"{section}_{key}"] = init_args[key]

        return summary
