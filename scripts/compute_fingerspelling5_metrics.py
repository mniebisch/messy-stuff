from lightning.pytorch.cli import LightningCLI

from fmp import datasets, models


def cli_main():
    cli = LightningCLI(
        model_class=models.Identity,
        datamodule_class=datasets.fingerspelling5.Fingerspelling5LandmarkDataModule,
    )


if __name__ == "__main__":
    cli_main()
    # TODO write yaml [X]
    # TODO add example to README [X]
    # TODO add metric callback [X]
    # TODO keep SCALE and NOT SCALE in mind! # maybe do afterwards and
    # read from file name or transform config
    # TODO apply everthing using dummy script
