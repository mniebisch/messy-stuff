from typing import List, Optional

import pandas as pd
import lightning as L
from sklearn import model_selection
from torch.utils import data as torch_data
from torch_geometric.transforms import BaseTransform

from .fingerspelling5 import Fingerspelling5Landmark


__all__ = ["Fingerspelling5LandmarkDataModule"]


class Fingerspelling5LandmarkDataModule(L.LightningDataModule):
    def __init__(
        self,
        fingerspelling5_csv: str,
        batch_size: int,
        train_transforms: Optional[BaseTransform] = None,
        valid_transforms: Optional[BaseTransform] = None,
    ) -> None:
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

            self.train_data = Fingerspelling5Landmark(
                train_data, transforms=self.hparams.train_transforms, filter_nans=True
            )
            self.valid_train_split = Fingerspelling5Landmark(
                train_data,
                transforms=self.hparams.valid_transforms,
                filter_nans=True,
                split="train",
            )
            self.valid_valid_split = Fingerspelling5Landmark(
                valid_data,
                transforms=self.hparams.valid_transforms,
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
