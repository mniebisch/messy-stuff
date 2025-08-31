from lightning.pytorch.cli import LightningCLI

from fmp import models


def cli_main():
    cli = LightningCLI(
        model_class=models.Identity,
    )


if __name__ == "__main__":
    cli_main()
