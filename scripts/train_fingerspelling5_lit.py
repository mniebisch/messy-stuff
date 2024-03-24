import pathlib

import torchvision
import lightning as L
import pandas as pd
from torch_geometric import transforms as pyg_transforms
from sklearn import model_selection
from torch.utils import data as torch_data

torchvision.disable_beta_transforms_warning()

from fmp import datasets, models


if __name__ == "__main__":
    lr = "blub"
    num_epochs = 20
    batch_size = 128

    # Load Data
    data_path = pathlib.Path(__file__).parent.parent / "data"
    # fingerspelling_landmark_csv = data_path / "fingerspelling5_singlehands.csv"
    fingerspelling_landmark_csv = data_path / "fingerspelling5_dummy_data.csv"
    landmark_data = pd.read_csv(fingerspelling_landmark_csv)

    # Create Random Group Split
    groups = landmark_data["person"]
    gss = model_selection.GroupShuffleSplit(n_splits=3, train_size=0.8)
    train_index, val_index = next(gss.split(None, None, groups))
    train_data = landmark_data.loc[train_index]
    valid_data = landmark_data.loc[val_index]

    # Setup Training Datset
    train_transforms = pyg_transforms.Compose(
        [
            pyg_transforms.NormalizeScale(),  # rethink order of transforms?
            pyg_transforms.RandomFlip(axis=0),
            pyg_transforms.RandomJitter(0.05),
            pyg_transforms.RandomRotate(degrees=20, axis=0),
            pyg_transforms.RandomRotate(degrees=20, axis=1),
            pyg_transforms.RandomRotate(degrees=20, axis=2),
        ]
    )
    train_dataset = datasets.fingerspelling5.Fingerspelling5Landmark(
        train_data, transforms=train_transforms, filter_nans=True
    )
    train_dataloader = torch_data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Setup Validation
    # # Validation Transforms
    valid_transforms = pyg_transforms.Compose(
        [
            pyg_transforms.NormalizeScale(),
        ]
    )
    # # Training Data Validation
    train_valid_dataset = datasets.fingerspelling5.Fingerspelling5Landmark(
        train_data, transforms=valid_transforms, filter_nans=True, split="train"
    )
    train_valid_dataloader = torch_data.DataLoader(
        train_valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    # # Validation Data Validation
    valid_dataset = datasets.fingerspelling5.Fingerspelling5Landmark(
        valid_data, transforms=valid_transforms, filter_nans=True, split="val"
    )
    valid_dataloader = torch_data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    model = models.LitMLP(
        input_dim=train_dataset.num_features,
        hidden_dim=10,
        output_dim=train_dataset.num_letters,
    )

    trainer = L.Trainer(
        max_epochs=num_epochs,
        log_every_n_steps=1,
        check_val_every_n_epoch=3,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=[valid_dataloader, train_valid_dataloader],
    )
