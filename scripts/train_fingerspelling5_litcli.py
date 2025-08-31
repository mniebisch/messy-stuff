import torch
from lightning.pytorch.cli import LightningCLI

from fmp.lit_tools.callbacks.aim import AimConfigWriter

torch.set_float32_matmul_precision("high")


def cli_main():
    cli = LightningCLI(
        save_config_callback=AimConfigWriter,
        save_config_kwargs={"logdir": "./config_logs"},
    )


if __name__ == "__main__":
    cli_main()
