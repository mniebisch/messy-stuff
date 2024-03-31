from lightning.pytorch.cli import LightningCLI

from fmp import datasets, models

def cli_main():
    cli = LightningCLI(
        model_class=models.LitMLP, 
        datamodule_class=datasets.fingerspelling5.Fingerspelling5LandmarkDataModule
    )

if __name__ == "__main__":
    cli_main()