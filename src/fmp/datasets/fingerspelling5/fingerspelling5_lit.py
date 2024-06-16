import pathlib
from typing import List, Optional, Tuple
import warnings

import pandas as pd
import lightning as L
from torch.utils import data as torch_data
from torch_geometric.transforms import BaseTransform
from numpy import typing as npt

from fmp.datasets import fingerspelling5


__all__ = ["Fingerspelling5LandmarkDataModule"]


class Fingerspelling5LandmarkDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        train_transforms: Optional[BaseTransform] = None,
        valid_transforms: Optional[BaseTransform] = None,
        predict_transforms: Optional[BaseTransform] = None,
        datasplit_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.fingerspelling5_csv = self.get_data_filename()

        self.validate_dataset_dir()

        if self.hparams.datasplit_file is not None:
            self.validate_datasplit_file()

    def setup(self, stage: str):
        filter_nans = True
        if stage == "fit":
            if self.hparams.datasplit_file is None:
                raise ValueError(
                    "Fit without 'datasplit_file' not possible. "
                    "Please provide 'datasplit_file'."
                )

            landmark_data = fingerspelling5.utils.read_csv(
                self.fingerspelling5_csv, filter_nans=filter_nans
            )

            self.validate_datasplit_data(landmark_data)
            # hm, can split file change between these line?
            train_index, val_index = self.load_datasplit_indices()

            train_data = landmark_data.loc[train_index]
            valid_data = landmark_data.loc[val_index]

            self.train_data = fingerspelling5.Fingerspelling5Landmark(
                train_data, transforms=self.hparams.train_transforms
            )
            self.valid_train_split = fingerspelling5.Fingerspelling5Landmark(
                train_data,
                transforms=self.hparams.valid_transforms,
                split="train",
            )
            self.valid_valid_split = fingerspelling5.Fingerspelling5Landmark(
                valid_data,
                transforms=self.hparams.valid_transforms,
                split="valid",
            )

        elif stage == "test":
            pass
        elif stage == "predict":
            landmark_data = fingerspelling5.utils.read_csv(
                self.fingerspelling5_csv, filter_nans=filter_nans
            )
            self.predict_data = fingerspelling5.Fingerspelling5Landmark(
                landmark_data,
                transforms=self.hparams.predict_transforms,
            )
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
        return torch_data.DataLoader(
            self.predict_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def validate_dataset_dir(self) -> None:
        if not self.fingerspelling5_csv.exists():
            raise ValueError(
                f"Expected dataset CSV '{str(self.fingerspelling5_csv)}' not found."
            )

        if not self.fingerspelling5_csv.is_file():
            raise ValueError(
                f"Dataset CSV '{str(self.fingerspelling5_csv)}' is not a file."
            )

        dataset_path = pathlib.Path(self.hparams.dataset_dir)
        csv_data_files = _extract_data_csv_files(dataset_path)

        if len(csv_data_files) > 1:
            warnings.warn(
                f"Multiple data CSVs found in {self.hparams.dataset_dir}. "
                "The dataset directory is expected to have only one data CSV. "
                "Please check directory to ensure correct data is used."
            )

    def validate_datasplit_file(self) -> None:
        datasplit_file = pathlib.Path(self.hparams.datasplit_file)

        if not datasplit_file.exists():
            raise ValueError(f"Invalid datasplit file '{str(datasplit_file)}'.")

    def validate_datasplit_data(self, landmark_data: pd.DataFrame) -> None:
        split_data = pd.read_csv(self.hparams.datasplit_file)

        if len(split_data) != len(landmark_data):
            raise ValueError("Landmark data and split data are not of same length.")

        if not all(landmark_data["person"] == split_data["person"]) or not all(
            landmark_data["letter"] == split_data["letter"]
        ):
            raise ValueError(
                "Sanity check between 'split_data' and 'landmark_data' shows "
                "that the order of rows has change. Please fix."
            )

        if (
            not "train" in split_data["split"].values
            or not "valid" in split_data["split"].values
        ):
            raise ValueError(
                "Expect the split identifiers 'train' and 'split' but didn't found them."
            )
        # check if other elements than 'train' or 'valid' are in split col?

    def load_datasplit_indices(self) -> Tuple[npt.NDArray, npt.NDArray]:
        split_data = pd.read_csv(self.hparams.datasplit_file)

        train_indices = split_data["split"] == "train"
        valid_indices = split_data["split"] == "valid"

        return train_indices.values, valid_indices.values

    def extract_dataset_name(self) -> str:
        dataset_path = pathlib.Path(self.hparams.dataset_dir)
        return dataset_path.name

    def get_data_filename(self) -> pathlib.Path:
        dataset_path = pathlib.Path(self.hparams.dataset_dir)
        dataset_name = self.extract_dataset_name()
        file_name = f"{dataset_name}.csv"
        return dataset_path / file_name


def _extract_data_csv_files(data_dir: pathlib.Path) -> List[pathlib.Path]:
    return [
        file
        for file in data_dir.glob("*")
        if file.name.casefold().endswith(".csv")
        and not file.name.casefold().startswith("split_")
    ]
