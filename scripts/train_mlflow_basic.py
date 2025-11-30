"""
Basic MLFlow training script with unified config callback.

This script demonstrates the standard usage of MLFlow logging with
the unified MLFlowConfigCallback, providing a clean and simple approach.
"""

import torch
from lightning.pytorch.cli import LightningCLI

from fmp.lit_tools.callbacks.mlflow import MLFlowConfigCallback

# # Patch torch.load to use weights_only=False by default
# _original_torch_load = torch.load


# def patched_load(*args, **kwargs):
#     kwargs.setdefault("weights_only", False)
#     return _original_torch_load(*args, **kwargs)


# torch.load = patched_load


torch.set_float32_matmul_precision("high")


def cli_main():
    """Run standard MLFlow CLI with unified config callback."""
    LightningCLI(
        save_config_callback=MLFlowConfigCallback,
        save_config_kwargs={
            "logdir": "./config_logs",
            "artifact_path": "configs",
            "log_as_artifact": True,
            "log_config_hash": False,  # Disabled - MLFlow run ID provides uniqueness
            "log_summary_tags": True,
        },
    )


if __name__ == "__main__":
    cli_main()
