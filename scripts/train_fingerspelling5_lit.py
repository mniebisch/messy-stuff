import pathlib

import torchvision
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor

torchvision.disable_beta_transforms_warning()

from fmp import datasets, models


if __name__ == "__main__":
    lr = 0.001
    num_epochs = 20
    batch_size = 128


    # Load Data
    data_path = pathlib.Path(__file__).parent.parent / "data"
    # fingerspelling_landmark_csv = data_path / "fingerspelling5_singlehands.csv"
    fingerspelling_landmark_csv = data_path / "fingerspelling5_dummy_data.csv"
    fingerspelling5 = datasets.fingerspelling5.Fingerspelling5LandmarkDataModule(
        fingerspelling_landmark_csv, 
        batch_size=batch_size,
    )

    model = models.LitMLP(
        input_dim=fingerspelling5.num_features,
        hidden_dim=10,
        output_dim=fingerspelling5.num_letters,
        learning_rate=lr,
        scheduler_T_max=num_epochs,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=num_epochs,
        log_every_n_steps=1,
        check_val_every_n_epoch=3,
        callbacks=[lr_monitor],
    )
    trainer.fit(
        model=model,
        datamodule=fingerspelling5,
    )
