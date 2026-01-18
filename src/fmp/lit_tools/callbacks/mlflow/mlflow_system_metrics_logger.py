"""Extended MLFlow Logger with system metrics support.

This module provides an extended MLFlowLogger that enables system metrics
logging (CPU, GPU, memory utilization) which is not available in the
standard Lightning MLFlowLogger.

The standard Lightning MLFlowLogger uses `MlflowClient.create_run()` which
doesn't support the `log_system_metrics` parameter. This extended logger
uses `mlflow.start_run()` instead to enable system metrics.

Example:
    >>> from fmp.lit_tools.callbacks.mlflow import MLFlowSystemMetricsLogger
    >>>
    >>> logger = MLFlowSystemMetricsLogger(
    ...     experiment_name="my_experiment",
    ...     tracking_uri="sqlite:///mlruns.db",
    ...     log_system_metrics=True,  # Enable CPU/GPU/memory tracking
    ... )
    >>> trainer = L.Trainer(logger=logger)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Literal

from lightning.pytorch.loggers import MLFlowLogger
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

__all__ = ["MLFlowSystemMetricsLogger"]

logger = logging.getLogger(__name__)

_MLFLOW_AVAILABLE = RequirementCache("mlflow>=1.0.0", "mlflow")


class MLFlowSystemMetricsLogger(MLFlowLogger):
    """Extended MLFlowLogger with system metrics support.

    This logger extends the standard Lightning MLFlowLogger to support
    system metrics logging (CPU/GPU utilization, memory usage, etc.).

    The key difference from the standard MLFlowLogger is that this logger
    uses `mlflow.start_run()` instead of `MlflowClient.create_run()`,
    which allows enabling system metrics tracking.

    System metrics include:
        - CPU utilization
        - GPU utilization (if available)
        - Memory usage
        - Disk I/O
        - Network I/O

    Note:
        System metrics logging requires MLflow >= 2.8.0 and may require
        additional dependencies (`pip install mlflow[system-metrics]`).

    Attributes:
        log_system_metrics: Whether system metrics logging is enabled.

    Example:
        >>> logger = MLFlowSystemMetricsLogger(
        ...     experiment_name="training",
        ...     tracking_uri="sqlite:///mlruns.db",
        ...     log_system_metrics=True,
        ... )
    """

    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        run_name: str | None = None,
        tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI"),
        tags: dict[str, Any] | None = None,
        save_dir: str | None = "./mlruns",
        log_model: Literal[True, False, "all"] = False,
        prefix: str = "",
        artifact_location: str | None = None,
        run_id: str | None = None,
        synchronous: bool | None = None,
        # New parameter for system metrics
        log_system_metrics: bool = False,
    ) -> None:
        """Initialize the extended MLFlow logger.

        Args:
            experiment_name: The name of the experiment.
            run_name: Name of the new run.
            tracking_uri: Address of local or remote tracking server.
            tags: Dictionary of tags for the experiment.
            save_dir: Path to local directory where MLflow runs get saved.
            log_model: Whether to log checkpoints as MLFlow artifacts.
            prefix: String to put at the beginning of metric keys.
            artifact_location: Location to store run artifacts.
            run_id: Run identifier to resume an existing run.
            synchronous: Whether to block on logging calls.
            log_system_metrics: Whether to enable system metrics logging
                (CPU, GPU, memory utilization). Requires MLflow >= 2.8.0.
        """
        # Store system metrics flag before calling parent __init__
        self._log_system_metrics = log_system_metrics
        self._system_metrics_initialized = False

        # Call parent constructor
        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            tags=tags,
            save_dir=save_dir,
            log_model=log_model,
            prefix=prefix,
            artifact_location=artifact_location,
            run_id=run_id,
            synchronous=synchronous,
        )

    @property
    def log_system_metrics(self) -> bool:
        """Whether system metrics logging is enabled."""
        return self._log_system_metrics

    @property
    @override
    def experiment(self) -> MlflowClient:
        """Get MLflow client, initializing with system metrics if enabled.

        This overrides the parent's experiment property to use mlflow.start_run()
        when system metrics are enabled.

        Returns:
            The MLflow tracking client.
        """
        import mlflow
        from mlflow.tracking import MlflowClient

        if self._initialized:
            return self._mlflow_client

        mlflow.set_tracking_uri(self._tracking_uri)

        # Handle resuming existing run
        if self._run_id is not None:
            run = self._mlflow_client.get_run(self._run_id)
            self._experiment_id = run.info.experiment_id

            if self._log_system_metrics and not self._system_metrics_initialized:
                # Start system metrics monitoring for existing run
                try:
                    mlflow.start_run(
                        run_id=self._run_id,
                        log_system_metrics=True,
                    )
                    self._system_metrics_initialized = True
                    logger.info("System metrics logging enabled for resumed run")
                except Exception as e:
                    logger.warning(f"Failed to enable system metrics: {e}")

            self._initialized = True
            return self._mlflow_client

        # Create or get experiment
        if self._experiment_id is None:
            expt = self._mlflow_client.get_experiment_by_name(self._experiment_name)
            if expt is not None and expt.lifecycle_stage != "deleted":
                self._experiment_id = expt.experiment_id
            else:
                logger.warning(
                    f"Experiment with name {self._experiment_name} not found. Creating it."
                )
                self._experiment_id = self._mlflow_client.create_experiment(
                    name=self._experiment_name,
                    artifact_location=self._artifact_location,
                )

        # Create new run
        if self._run_id is None:
            # Prepare tags
            tags = self.tags or {}
            if self._run_name is not None:
                from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

                tags[MLFLOW_RUN_NAME] = self._run_name

            if self._log_system_metrics:
                # Use mlflow.start_run() to enable system metrics
                try:
                    mlflow.set_experiment(experiment_id=self._experiment_id)
                    active_run = mlflow.start_run(
                        run_name=self._run_name,
                        tags=tags,
                        log_system_metrics=True,
                    )
                    self._run_id = active_run.info.run_id
                    self._system_metrics_initialized = True
                    logger.info(
                        f"Created MLflow run with system metrics: {self._run_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to create run with system metrics, falling back: {e}"
                    )
                    # Fall back to standard run creation
                    self._create_run_without_system_metrics(tags)
            else:
                # Standard run creation (same as parent)
                self._create_run_without_system_metrics(tags)

        self._initialized = True
        return self._mlflow_client

    def _create_run_without_system_metrics(self, tags: dict[str, Any]) -> None:
        """Create a run without system metrics (fallback/standard behavior).

        Args:
            tags: Tags to set on the run.
        """
        assert self._experiment_id is not None, "Experiment ID must be set"

        def _resolve_tags(tag_dict: dict[str, Any] | None) -> dict[str, str] | None:
            """Resolve tags to string values."""
            if tag_dict is None:
                return None
            return {k: str(v) for k, v in tag_dict.items()}

        run = self._mlflow_client.create_run(
            experiment_id=self._experiment_id, tags=_resolve_tags(tags)
        )
        self._run_id = run.info.run_id

    def finalize(self, status: str = "success") -> None:  # type: ignore[override]
        """Finalize the logger, ending the MLflow run if needed.

        Args:
            status: Final status of the training run.
        """
        import mlflow

        # End the run if we started it with mlflow.start_run()
        if self._system_metrics_initialized:
            try:
                mlflow.end_run()
                logger.debug("Ended MLflow run with system metrics")
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")

        super().finalize(status)
