from typing import List

import pandas as pd
import lightning as L
from sklearn import model_selection
from torch_geometric import transforms as pyg_transforms
from torch.utils import data as torch_data

from .fingerspelling5 import Fingerspelling5Landmark


__all__ = ["Fingerspelling5LandmarkDataModule"]


class Fingerspelling5LandmarkDataModule(L.LightningDataModule):
    def __init__(self, fingerspelling5_csv: str, batch_size: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage: str):
        if stage == "fit":
            landmark_data = pd.read_csv(self.hparams.fingerspelling5_csv)
            groups = landmark_data["person"]
            gss = model_selection.GroupShuffleSplit(n_splits=3, train_size=0.8)
            train_index, val_index = next(gss.split(None, None, groups))
            train_data = landmark_data.loc[train_index]
            valid_data = landmark_data.loc[val_index]

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
            valid_transforms = pyg_transforms.Compose(
                [
                    pyg_transforms.NormalizeScale(),
                ]
            )

            self.train_data = Fingerspelling5Landmark(
                train_data, transforms=train_transforms, filter_nans=True
            )
            self.valid_train_split = Fingerspelling5Landmark(
                train_data, 
                transforms=valid_transforms, 
                filter_nans=True,
                split="train",
            )
            self.valid_valid_split = Fingerspelling5Landmark(
                valid_data, 
                transforms=valid_transforms, 
                filter_nans=True,
                split="valid",
            )

        elif stage == "test":
            pass
        elif stage == "predict":
            pass
        else:
            pass

    def prepare_data(self):
        pass

    def train_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> List[torch_data.DataLoader]:
        train_loader = torch_data.DataLoader(
            self.valid_train_split,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
        )
        valid_loader = torch_data.DataLoader(
            self.valid_valid_split,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
        )
        return [train_loader, valid_loader]

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self):
        raise NotImplementedError
