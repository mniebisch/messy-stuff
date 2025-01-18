import pathlib
import warnings
from typing import List, Optional, Tuple, Union

import albumentations as A
import lightning as L
import pandas as pd
from numpy import typing as npt
from torch.utils import data as torch_data
from torch_geometric.transforms import BaseTransform

from fmp.datasets import fingerspelling5

__all__ = ["Fingerspelling5ImageDataModule", "Fingerspelling5LandmarkDataModule"]


class Fingerspelling5LandmarkDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        num_dataloader_workers: int = 0,
        train_transforms: Optional[BaseTransform] = None,
        valid_transforms: Optional[BaseTransform] = None,
        predict_transforms: Optional[BaseTransform] = None,
        datasplit_file: Optional[str] = None,
        dataquality_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.predict_transforms = predict_transforms
        self.datasplit_file = datasplit_file
        self.dataquality_file = dataquality_file

        self.fingerspelling5_csv = self.get_data_filename()
        self.validate_dataset_dir()

        if self.datasplit_file is not None:
            self.validate_datasplit_file(self.datasplit_file)

        if self.dataquality_file is not None:
            self.validate_dataquality_file(self.dataquality_file)

    def setup(self, stage: str):
        filter_nans = True
        if stage == "fit":
            if self.datasplit_file is None:
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

            if self.dataquality_file is not None:
                self.validate_dataquality_data(landmark_data)
                quality_indices = pd.read_csv(self.dataquality_file)["is_corrupted"]

                train_index = train_index & ~quality_indices.values
                val_index = val_index & ~quality_indices.values

            train_data = landmark_data.loc[train_index]
            valid_data = landmark_data.loc[val_index]

            self.train_data = fingerspelling5.Fingerspelling5Landmark(
                train_data, transforms=self.train_transforms
            )
            self.valid_train_split = fingerspelling5.Fingerspelling5Landmark(
                train_data,
                transforms=self.valid_transforms,
                split="train",
            )
            self.valid_valid_split = fingerspelling5.Fingerspelling5Landmark(
                valid_data,
                transforms=self.valid_transforms,
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
                transforms=self.predict_transforms,
            )
        else:
            pass

    def prepare_data(self):
        pass

    def train_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_dataloader_workers,
        )

    def val_dataloader(self) -> List[torch_data.DataLoader]:
        train_loader = torch_data.DataLoader(
            self.valid_train_split,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_dataloader_workers,
        )
        valid_loader = torch_data.DataLoader(
            self.valid_valid_split,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_dataloader_workers,
        )
        return [train_loader, valid_loader]

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self):
        return torch_data.DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_dataloader_workers,
        )

    def validate_dataset_dir(
        self,
    ) -> None:
        if not self.fingerspelling5_csv.exists():
            raise ValueError(
                f"Expected dataset CSV '{str(self.fingerspelling5_csv)}' not found."
            )

        if not self.fingerspelling5_csv.is_file():
            raise ValueError(
                f"Dataset CSV '{str(self.fingerspelling5_csv)}' is not a file."
            )

        dataset_path = pathlib.Path(self.dataset_dir)
        csv_data_files = _extract_data_csv_files(dataset_path)

        if len(csv_data_files) > 1:
            warnings.warn(
                f"Multiple data CSVs found in {self.dataset_dir}. "
                "The dataset directory is expected to have only one data CSV. "
                "Please check directory to ensure correct data is used."
            )

    @staticmethod
    def validate_datasplit_file(datasplit_file: str) -> None:
        if not pathlib.Path(datasplit_file).exists():
            raise ValueError(f"Invalid datasplit file '{datasplit_file}'.")

    def validate_datasplit_data(self, landmark_data: pd.DataFrame) -> None:
        split_data = pd.read_csv(self.datasplit_file)

        if len(split_data) != len(landmark_data):
            raise ValueError("Landmark data and split data are not of same length.")

        if not all(landmark_data["img_file"] == split_data["img_file"]):
            raise ValueError(
                "Sources for landmark data and split data are not the same or "
                "the order has changed."
            )

        if (
            "train" not in split_data["split"].values
            or "valid" not in split_data["split"].values
        ):
            raise ValueError(
                "Expect the split identifiers 'train' and 'split' but didn't found them."
            )
        # check if other elements than 'train' or 'valid' are in split col?

    def load_datasplit_indices(self) -> Tuple[npt.NDArray, npt.NDArray]:
        split_data = pd.read_csv(self.datasplit_file)

        train_indices = split_data["split"] == "train"
        valid_indices = split_data["split"] == "valid"

        return train_indices.values, valid_indices.values

    def extract_dataset_name(self) -> str:
        dataset_path = pathlib.Path(self.dataset_dir)
        return dataset_path.name

    def get_data_filename(self) -> pathlib.Path:
        dataset_path = pathlib.Path(self.dataset_dir)
        dataset_name = self.extract_dataset_name()
        file_name = f"{dataset_name}.csv"
        return dataset_path / file_name

    def validate_dataquality_file(self, dataquality_file: str) -> None:
        if not pathlib.Path(dataquality_file).exists():
            raise ValueError(f"Invalid dataquality file '{dataquality_file}'.")

    def validate_dataquality_data(self, landmark_data: pd.DataFrame) -> None:
        dataquality_data = pd.read_csv(self.dataquality_file)

        if len(dataquality_data) != len(landmark_data):
            raise ValueError(
                "Landmark data and dataquality data are not of same length."
            )

        if not all(landmark_data["img_file"] == dataquality_data["img_file"]):
            raise ValueError(
                "Sources for landmark data and dataquality data are not the same or "
                "the order has changed."
            )

        if "is_corrupted" not in dataquality_data.columns:
            raise ValueError(
                "Expect the column 'is_corrupted' in dataquality data but didn't found it."
            )


class Fingerspelling5ImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        datasplit_file: str,
        images_data_dir: str,
        batch_size: int,
        num_dataloader_workers: int = 0,
        train_transforms: Optional[Union[A.BaseCompose, A.BasicTransform]] = None,
        valid_transforms: Optional[Union[A.BaseCompose, A.BasicTransform]] = None,
        predict_transforms: Optional[Union[A.BaseCompose, A.BasicTransform]] = None,
        dataquality_file: Optional[str] = None,
    ) -> None:
        # TODO add validation if required
        # TODO maybe find better name than datasplit file? predict case!?
        # TODO read dataquality file and check if it is valid
        super().__init__()
        self.save_hyperparameters()

        self.images_data_dir = images_data_dir
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.predict_transforms = predict_transforms
        self.datasplit_file = datasplit_file
        self.dataquality_file = dataquality_file

    def setup(self, stage: str) -> None:
        if stage == "fit":
            fingerspelling5_data = pd.read_csv(self.datasplit_file)

            self.train_data = fingerspelling5.Fingerspelling5Image(
                fingerspelling5_data.loc[
                    fingerspelling5_data["split"] == "train"
                ].reset_index(drop=True),
                pathlib.Path(self.images_data_dir),
                transforms=self.train_transforms,
            )

            self.valid_train_data = fingerspelling5.Fingerspelling5Image(
                fingerspelling5_data.loc[
                    fingerspelling5_data["split"] == "train"
                ].reset_index(drop=True),
                pathlib.Path(self.images_data_dir),
                transforms=self.valid_transforms,
                split="train",
            )

            self.valid_valid_data = fingerspelling5.Fingerspelling5Image(
                fingerspelling5_data.loc[
                    fingerspelling5_data["split"] == "valid"
                ].reset_index(drop=True),
                pathlib.Path(self.images_data_dir),
                transforms=self.valid_transforms,
                split="valid",
            )
        elif stage == "test":
            pass
        elif stage == "predict":
            fingerspelling5_data = pd.read_csv(self.datasplit_file)

            self.predict_data = fingerspelling5.Fingerspelling5Image(
                fingerspelling5_data,
                pathlib.Path(self.images_data_dir),
                transforms=self.predict_transforms,
            )
        else:
            pass

    def prepare_data(self) -> None:
        pass

    def train_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_dataloader_workers,
        )

    def val_dataloader(self) -> List[torch_data.DataLoader]:
        train_loader = torch_data.DataLoader(
            self.valid_train_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_dataloader_workers,
        )
        valid_loader = torch_data.DataLoader(
            self.valid_valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_dataloader_workers,
        )
        return [train_loader, valid_loader]

    def test_dataloader(self) -> None:
        raise NotImplementedError

    def predict_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_dataloader_workers,
        )


def _extract_data_csv_files(data_dir: pathlib.Path) -> List[pathlib.Path]:
    return [
        file
        for file in data_dir.glob("*")
        if file.name.casefold().endswith(".csv")
        and not file.name.casefold().startswith("split_")
    ]
