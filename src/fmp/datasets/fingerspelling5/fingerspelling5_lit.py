import pathlib
import warnings
from typing import Any, Dict, List, Optional, Tuple

import kornia as K
import lightning as L
import numpy as np
import pandas as pd
import torch
import tqdm
from kornia.augmentation.base import _AugmentationBase
from numpy import typing as npt
from torch.utils import data as torch_data
from torch_geometric.transforms import BaseTransform
from torchvision.transforms import v2

from fmp.datasets import fingerspelling5
from fmp.transforms import PadToSize

__all__ = [
    "Fingerspelling5ImageDataModule",
    "Fingerspelling5ImageInmemoryDataModule",
    "Fingerspelling5LandmarkDataModule",
]


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
        self.dataset_name = pathlib.Path(dataset_dir).name

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

            split_data = pd.read_csv(self.datasplit_file)
            validate_datasplit_data(landmark_data, split_data)
            # hm, can split file change between these line?
            train_index, val_index = load_datasplit_indices(split_data)

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


def validate_additional_valid_keys(
    image_files: Optional[Dict[str, str]],
    transforms: Optional[Dict[str, List[_AugmentationBase]]],
) -> None:
    """Validate that additional validation datasets and transforms are properly configured."""
    # Both must be provided together or both None
    if (image_files is None) != (transforms is None):
        raise ValueError(
            "Both 'additional_valid_images_files' and "
            "'additional_valid_image_transform' must be provided together, or both None."
        )

    # If both are None, validation passes
    if image_files is None and transforms is None:
        return

    # Check that keys match
    image_keys = set(image_files.keys())
    transform_keys = set(transforms.keys())

    if image_keys != transform_keys:
        raise ValueError(
            f"Keys mismatch between additional datasets and transforms. "
            f"Image files keys: {image_keys}, Transform keys: {transform_keys}"
        )

    # Validate file paths exist
    for dataset_name, csv_path in image_files.items():
        if not pathlib.Path(csv_path).exists():
            raise ValueError(f"Additional dataset CSV file not found: {csv_path}")

    # Validate no conflicts with reserved split names
    reserved_names = {"train_split", "valid_split", "train", "valid"}
    conflicting_names = image_keys.intersection(reserved_names)
    if conflicting_names:
        raise ValueError(
            f"Additional dataset names conflict with reserved names: {conflicting_names}. "
            f"Reserved names are: {reserved_names}"
        )


class Fingerspelling5ImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        images_data_dir: str,
        batch_size: int,
        num_dataloader_workers: int = 0,
        train_transforms: Optional[v2.Transform] = None,
        kornia_train_transforms: Optional[List[_AugmentationBase]] = None,
        kornia_train_transform_kwargs: Optional[Dict[str, Any]] = None,
        kornia_valid_transform_kwargs: Optional[Dict[str, Any]] = None,
        kornia_predict_transform_kwargs: Optional[Dict[str, Any]] = None,
        kornia_valid_transforms: Optional[List[_AugmentationBase]] = None,
        kornia_test_transforms: Optional[List[_AugmentationBase]] = None,
        kornia_test_transform_kwargs: Optional[Dict[str, Any]] = None,
        kornia_predict_transforms: Optional[List[_AugmentationBase]] = None,
        valid_transforms: Optional[v2.Transform] = None,
        predict_transforms: Optional[v2.Transform] = None,
        test_transforms: Optional[v2.Transform] = None,
        datasplit_file: Optional[str] = None,
        dataquality_file: Optional[str] = None,
        cpu_batch_transforms: Optional[List[_AugmentationBase]] = None,
        gpu_batch_transforms: Optional[List[_AugmentationBase]] = None,
        cpu_batch_transform_kwargs: Optional[Dict[str, Any]] = None,
        gpu_batch_transform_kwargs: Optional[Dict[str, Any]] = None,
        additional_valid_images_files: Optional[Dict[str, str]] = None,
        additional_valid_image_transform: Optional[
            Dict[str, List[_AugmentationBase]]
        ] = None,
        letters: Optional[List[str]] = None,
    ) -> None:

        # TODO add validation if required
        # TODO maybe find better name than datasplit file? predict case!?
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "train_transforms",
                "valid_transforms",
                "test_transforms",
                "predict_transforms",
                "kornia_train_transforms",
                "kornia_valid_transforms",
                "kornia_test_transforms",
                "kornia_predict_transforms",
                "cpu_batch_transforms",
                "gpu_batch_transforms",
                "additional_valid_image_transform",
            ],
        )

        self.dataset_dir = dataset_dir
        self.images_data_dir = images_data_dir
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.test_transforms = test_transforms
        self.predict_transforms = predict_transforms

        kornia_train_transform_kwargs = (
            {}
            if kornia_train_transform_kwargs is None
            else kornia_train_transform_kwargs
        )
        kornia_valid_transform_kwargs = (
            {}
            if kornia_valid_transform_kwargs is None
            else kornia_valid_transform_kwargs
        )
        kornia_test_transform_kwargs = (
            {} if kornia_test_transform_kwargs is None else kornia_test_transform_kwargs
        )
        kornia_predict_transform_kwargs = (
            {}
            if kornia_predict_transform_kwargs is None
            else kornia_predict_transform_kwargs
        )
        self.kornia_train_transforms = (
            K.augmentation.AugmentationSequential(
                *kornia_train_transforms, **kornia_train_transform_kwargs
            )
            if kornia_train_transforms
            else None
        )
        self.kornia_valid_transforms = (
            K.augmentation.AugmentationSequential(
                *kornia_valid_transforms, **kornia_valid_transform_kwargs
            )
            if kornia_valid_transforms
            else None
        )
        self.kornia_test_transforms = (
            K.augmentation.AugmentationSequential(
                *kornia_test_transforms, **kornia_test_transform_kwargs
            )
            if kornia_test_transforms
            else None
        )
        self.kornia_predict_transforms = (
            K.augmentation.AugmentationSequential(
                *kornia_predict_transforms, **kornia_predict_transform_kwargs
            )
            if kornia_predict_transforms
            else None
        )

        cpu_batch_transform_kwargs = (
            {} if cpu_batch_transform_kwargs is None else cpu_batch_transform_kwargs
        )
        gpu_batch_transform_kwargs = (
            {} if gpu_batch_transform_kwargs is None else gpu_batch_transform_kwargs
        )

        self.cpu_batch_transforms = (
            K.augmentation.AugmentationSequential(
                *cpu_batch_transforms, **cpu_batch_transform_kwargs
            )
            if cpu_batch_transforms
            else None
        )
        self.gpu_batch_transforms = (
            K.augmentation.AugmentationSequential(
                *gpu_batch_transforms, **gpu_batch_transform_kwargs
            )
            if gpu_batch_transforms
            else None
        )

        self.datasplit_file = datasplit_file
        self.dataquality_file = dataquality_file
        self.dataset_name = pathlib.Path(dataset_dir).name

        self.image_files_csv = self.get_image_files_csv()

        validate_additional_valid_keys(
            additional_valid_images_files, additional_valid_image_transform
        )

        self.additional_valid_images_files = (
            additional_valid_images_files
            if additional_valid_images_files is not None
            else {}
        )
        self.additional_valid_image_transform = (
            additional_valid_image_transform
            if additional_valid_image_transform is not None
            else {}
        )
        self.letters = letters

    def setup(self, stage: str) -> None:
        if self.gpu_batch_transforms is not None:
            self.gpu_batch_transforms.to(self.trainer.model.device)

        if stage == "fit":
            fingerspelling5_image_files = pd.read_csv(self.image_files_csv)
            if self.datasplit_file is None:
                raise ValueError(
                    "Fit without 'datasplit_file' not possible. "
                    "Please provide 'datasplit_file'."
                )
            split_data = pd.read_csv(self.datasplit_file)
            validate_datasplit_data(fingerspelling5_image_files, split_data)
            train_index, valid_index = load_datasplit_indices(split_data)

            if self.dataquality_file is not None:
                dataquality_data = pd.read_csv(self.dataquality_file)
                quality_indices = dataquality_data["is_corrupted"]

                train_index = train_index & ~quality_indices.values
                valid_index = valid_index & ~quality_indices.values

            train_data = fingerspelling5_image_files.loc[train_index].reset_index(
                drop=True
            )
            valid_data = fingerspelling5_image_files.loc[valid_index].reset_index(
                drop=True
            )

            self.train_data = fingerspelling5.Fingerspelling5Image(
                train_data,
                pathlib.Path(self.images_data_dir),
                tv_transforms=self.train_transforms,
                kornia_transforms=self.kornia_train_transforms,
                letters=self.letters,
            )

            self.valid_train_data = fingerspelling5.Fingerspelling5Image(
                train_data,
                pathlib.Path(self.images_data_dir),
                tv_transforms=self.valid_transforms,
                kornia_transforms=self.kornia_valid_transforms,
                split="train_split",
                letters=self.letters,
            )

            self.valid_valid_data = fingerspelling5.Fingerspelling5Image(
                valid_data,
                pathlib.Path(self.images_data_dir),
                tv_transforms=self.valid_transforms,
                kornia_transforms=self.kornia_valid_transforms,
                split="valid_split",
                letters=self.letters,
            )

            # Create additional validation datasets
            self.additional_valid_datasets = {}
            for dataset_name, csv_path in self.additional_valid_images_files.items():
                # Read the additional dataset
                additional_data = pd.read_csv(csv_path)

                # Create Kornia transforms for this dataset
                transform_list = self.additional_valid_image_transform[dataset_name]
                kornia_transforms = (
                    K.augmentation.AugmentationSequential(
                        *transform_list, data_keys=["image"]
                    )
                    if transform_list
                    else None
                )

                # Create the dataset with the appropriate split name
                self.additional_valid_datasets[dataset_name] = (
                    fingerspelling5.Fingerspelling5Image(
                        additional_data,
                        pathlib.Path(self.images_data_dir),
                        kornia_transforms=kornia_transforms,
                        split=dataset_name,  # Use dataset name as split identifier
                        letters=self.letters,
                    )
                )
        elif stage == "test":
            fingerspelling5_image_files = pd.read_csv(self.image_files_csv)

            self.test_data = fingerspelling5.Fingerspelling5Image(
                fingerspelling5_image_files,
                pathlib.Path(self.images_data_dir),
                tv_transforms=self.test_transforms,
                kornia_transforms=self.kornia_test_transforms,
                letters=self.letters,
            )
        elif stage == "predict":
            fingerspelling5_image_files = pd.read_csv(self.image_files_csv)

            self.predict_data = fingerspelling5.Fingerspelling5Image(
                fingerspelling5_image_files,
                pathlib.Path(self.images_data_dir),
                tv_transforms=self.predict_transforms,
                kornia_transforms=self.kornia_predict_transforms,
                letters=self.letters,
            )
        elif stage == "validate":
            fingerspelling5_image_files = pd.read_csv(self.image_files_csv)
            if self.datasplit_file is None:
                raise ValueError(
                    "Fit without 'datasplit_file' not possible. "
                    "Please provide 'datasplit_file'."
                )
            split_data = pd.read_csv(self.datasplit_file)
            validate_datasplit_data(fingerspelling5_image_files, split_data)
            train_index, valid_index = load_datasplit_indices(split_data)

            if self.dataquality_file is not None:
                dataquality_data = pd.read_csv(self.dataquality_file)
                quality_indices = dataquality_data["is_corrupted"]

                train_index = train_index & ~quality_indices.values
                valid_index = valid_index & ~quality_indices.values

            train_data = fingerspelling5_image_files.loc[train_index].reset_index(
                drop=True
            )
            valid_data = fingerspelling5_image_files.loc[valid_index].reset_index(
                drop=True
            )

            self.valid_train_data = fingerspelling5.Fingerspelling5Image(
                train_data,
                pathlib.Path(self.images_data_dir),
                transforms=self.valid_transforms,
                split="train_split",
                letters=self.letters,
            )

            self.valid_valid_data = fingerspelling5.Fingerspelling5Image(
                valid_data,
                pathlib.Path(self.images_data_dir),
                transforms=self.valid_transforms,
                split="valid_split",
                letters=self.letters,
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
            persistent_workers=True,
            pin_memory=True,
            # prefetch_factor=1,
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
        # Base validation dataloaders
        dataloaders = [train_loader, valid_loader]

        # Additional validation datasets
        if hasattr(self, "additional_valid_datasets"):
            for dataset_name, dataset in self.additional_valid_datasets.items():
                additional_loader = torch_data.DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=self.num_dataloader_workers,
                )
                dataloaders.append(additional_loader)

        return dataloaders

    def test_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_dataloader_workers,
        )

    def predict_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_dataloader_workers,
        )

    def get_image_files_csv(self) -> pathlib.Path:
        dataset_path = pathlib.Path(self.dataset_dir)
        return dataset_path / "image_files.csv"

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training and self.cpu_batch_transforms is not None:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
                images = self.cpu_batch_transforms(images)
                return images, labels
            elif isinstance(batch, torch.Tensor):
                return self.cpu_batch_transforms(batch)
            else:
                raise TypeError(
                    f"Unsupported batch type: {type(batch)}. Expected list, tuple, or Tensor."
                )
        else:
            return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training and self.gpu_batch_transforms is not None:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
                images = self.gpu_batch_transforms(images)
                return images, labels
            elif isinstance(batch, torch.Tensor):
                return self.gpu_batch_transforms(batch)
            else:
                raise TypeError(
                    f"Unsupported batch type: {type(batch)}. Expected list, tuple, or Tensor."
                )
        else:
            return batch


class Fingerspelling5ImageInmemoryDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        images_data_dir: str,
        batch_size: int,
        num_dataloader_workers: int = 0,
        train_transforms: Optional[v2.Transform] = None,
        kornia_train_transforms: Optional[List[_AugmentationBase]] = None,
        kornia_train_transform_kwargs: Optional[Dict[str, Any]] = None,
        kornia_valid_transform_kwargs: Optional[Dict[str, Any]] = None,
        kornia_valid_transforms: Optional[List[_AugmentationBase]] = None,
        kornia_test_transform_kwargs: Optional[Dict[str, Any]] = None,
        kornia_test_transforms: Optional[List[_AugmentationBase]] = None,
        kornia_predict_transform_kwargs: Optional[Dict[str, Any]] = None,
        kornia_predict_transforms: Optional[List[_AugmentationBase]] = None,
        valid_transforms: Optional[v2.Transform] = None,
        test_transforms: Optional[v2.Transform] = None,
        predict_transforms: Optional[v2.Transform] = None,
        datasplit_file: Optional[str] = None,
        dataquality_file: Optional[str] = None,
    ) -> None:
        # TODO add validation if required
        # TODO maybe find better name than datasplit file? predict case!?
        super().__init__()
        self.save_hyperparameters(
            ignore=["train_transforms", "valid_transforms", "predict_transforms"]
        )

        self.dataset_dir = dataset_dir
        self.images_data_dir = images_data_dir
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.test_transforms = test_transforms
        self.predict_transforms = predict_transforms

        kornia_train_transform_kwargs = (
            {}
            if kornia_train_transform_kwargs is None
            else kornia_train_transform_kwargs
        )
        kornia_valid_transform_kwargs = (
            {}
            if kornia_valid_transform_kwargs is None
            else kornia_valid_transform_kwargs
        )
        kornia_test_transform_kwargs = (
            {} if kornia_test_transform_kwargs is None else kornia_test_transform_kwargs
        )
        kornia_predict_transform_kwargs = (
            {}
            if kornia_predict_transform_kwargs is None
            else kornia_predict_transform_kwargs
        )
        self.kornia_train_transforms = (
            K.augmentation.AugmentationSequential(
                *kornia_train_transforms, **kornia_train_transform_kwargs
            )
            if kornia_train_transforms
            else None
        )
        self.kornia_valid_transforms = (
            K.augmentation.AugmentationSequential(
                *kornia_valid_transforms, **kornia_valid_transform_kwargs
            )
            if kornia_valid_transforms
            else None
        )
        self.kornia_test_transforms = (
            K.augmentation.AugmentationSequential(
                *kornia_test_transforms, **kornia_test_transform_kwargs
            )
            if kornia_test_transforms
            else None
        )
        self.kornia_predict_transforms = (
            K.augmentation.AugmentationSequential(
                *kornia_predict_transforms, **kornia_predict_transform_kwargs
            )
            if kornia_predict_transforms
            else None
        )

        self.datasplit_file = datasplit_file
        self.dataquality_file = dataquality_file
        self.dataset_name = pathlib.Path(dataset_dir).name

        self.image_files_csv = self.get_image_files_csv()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            fingerspelling5_image_files = pd.read_csv(self.image_files_csv)
            if self.datasplit_file is None:
                raise ValueError(
                    "Fit without 'datasplit_file' not possible. "
                    "Please provide 'datasplit_file'."
                )
            split_data = pd.read_csv(self.datasplit_file)
            validate_datasplit_data(fingerspelling5_image_files, split_data)
            train_index, valid_index = load_datasplit_indices(split_data)

            if self.dataquality_file is not None:
                dataquality_data = pd.read_csv(self.dataquality_file)
                quality_indices = dataquality_data["is_corrupted"]

                train_index = train_index & ~quality_indices.values
                valid_index = valid_index & ~quality_indices.values

            train_data = fingerspelling5_image_files.loc[train_index].reset_index(
                drop=True
            )
            valid_data = fingerspelling5_image_files.loc[valid_index].reset_index(
                drop=True
            )

            images_data_dir = pathlib.Path(self.images_data_dir)
            train_images = [
                K.io.load_image(
                    images_data_dir / filename,
                    desired_type=K.io.ImageLoadType.RGB8,
                    device="cpu",
                )
                for filename in tqdm.tqdm(train_data["img_file"])
            ]
            train_image_sizes = np.stack(
                [np.array(image.shape[1:]) for image in train_images]
            )
            train_labels = train_data["letter"].values
            train_max_height, train_max_width = np.max(train_image_sizes, axis=0)
            train_padder = PadToSize(
                size=(train_max_height, train_max_width),
                fill=0,
            )

            train_images = [train_padder(image) for image in train_images]
            train_images = torch.stack(train_images)

            # self.train_data = fingerspelling5.Fingerspelling5Image(
            #     train_data,
            #     pathlib.Path(self.images_data_dir),
            #     tv_transforms=self.train_transforms,
            #     kornia_transforms=self.kornia_train_transforms,
            # )

            self.train_data = fingerspelling5.Fingerspelling5ImageInmemory(
                train_images,
                train_labels,
                tv_transforms=self.train_transforms,
                kornia_transforms=self.kornia_train_transforms,
            )

            valid_images = [
                K.io.load_image(
                    images_data_dir / filename,
                    desired_type=K.io.ImageLoadType.RGB8,
                    device="cpu",
                )
                for filename in tqdm.tqdm(valid_data["img_file"])
            ]
            valid_image_sizes = np.stack(
                [np.array(image.shape[1:]) for image in valid_images]
            )
            valid_labels = valid_data["letter"].values
            valid_max_height, valid_max_width = np.max(valid_image_sizes, axis=0)
            valid_padder = PadToSize(
                size=(valid_max_height, valid_max_width),
                fill=0,
            )
            valid_images = [valid_padder(image) for image in valid_images]
            valid_images = torch.stack(valid_images)

            # self.valid_train_data = fingerspelling5.Fingerspelling5Image(
            #     train_data,
            #     pathlib.Path(self.images_data_dir),
            #     tv_transforms=self.valid_transforms,
            #     kornia_transforms=self.kornia_valid_transforms,
            #     split="train",
            # )
            self.valid_train_data = fingerspelling5.Fingerspelling5ImageInmemory(
                valid_images,
                valid_labels,
                tv_transforms=self.valid_transforms,
                kornia_transforms=self.kornia_valid_transforms,
            )

            # self.valid_valid_data = fingerspelling5.Fingerspelling5Image(
            #     valid_data,
            #     pathlib.Path(self.images_data_dir),
            #     tv_transforms=self.valid_transforms,
            #     kornia_transforms=self.kornia_valid_transforms,
            #     split="valid",
            # )
            self.valid_valid_data = fingerspelling5.Fingerspelling5ImageInmemory(
                valid_images,
                valid_labels,
                tv_transforms=self.valid_transforms,
                kornia_transforms=self.kornia_valid_transforms,
            )
        elif stage == "test":
            fingerspelling5_image_files = pd.read_csv(self.image_files_csv)

            self.test_data = fingerspelling5.Fingerspelling5Image(
                fingerspelling5_image_files,
                pathlib.Path(self.images_data_dir),
                tv_transforms=self.test_transforms,
                kornia_transforms=self.kornia_test_transforms,
            )

        elif stage == "predict":
            fingerspelling5_image_files = pd.read_csv(self.image_files_csv)

            self.predict_data = fingerspelling5.Fingerspelling5Image(
                fingerspelling5_image_files,
                pathlib.Path(self.images_data_dir),
                tv_transforms=self.predict_transforms,
                kornia_transforms=self.kornia_predict_transforms,
            )
        elif stage == "validate":
            fingerspelling5_image_files = pd.read_csv(self.image_files_csv)
            if self.datasplit_file is None:
                raise ValueError(
                    "Fit without 'datasplit_file' not possible. "
                    "Please provide 'datasplit_file'."
                )
            split_data = pd.read_csv(self.datasplit_file)
            validate_datasplit_data(fingerspelling5_image_files, split_data)
            train_index, valid_index = load_datasplit_indices(split_data)

            if self.dataquality_file is not None:
                dataquality_data = pd.read_csv(self.dataquality_file)
                quality_indices = dataquality_data["is_corrupted"]

                train_index = train_index & ~quality_indices.values
                valid_index = valid_index & ~quality_indices.values

            train_data = fingerspelling5_image_files.loc[train_index].reset_index(
                drop=True
            )
            valid_data = fingerspelling5_image_files.loc[valid_index].reset_index(
                drop=True
            )

            self.valid_train_data = fingerspelling5.Fingerspelling5Image(
                train_data,
                pathlib.Path(self.images_data_dir),
                transforms=self.valid_transforms,
                split="train",
            )

            self.valid_valid_data = fingerspelling5.Fingerspelling5Image(
                valid_data,
                pathlib.Path(self.images_data_dir),
                transforms=self.valid_transforms,
                split="valid",
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
            persistent_workers=True,
            pin_memory=True,
            # prefetch_factor=1,
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
        # Base validation dataloaders
        dataloaders = [train_loader, valid_loader]

        # Additional validation datasets
        if hasattr(self, "additional_valid_datasets"):
            for dataset_name, dataset in self.additional_valid_datasets.items():
                additional_loader = torch_data.DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=self.num_dataloader_workers,
                )
                dataloaders.append(additional_loader)

        return dataloaders

    def test_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_dataloader_workers,
        )

    def predict_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_dataloader_workers,
        )

    def get_image_files_csv(self) -> pathlib.Path:
        dataset_path = pathlib.Path(self.dataset_dir)
        return dataset_path / "image_files.csv"


def _extract_data_csv_files(data_dir: pathlib.Path) -> List[pathlib.Path]:
    return [
        file
        for file in data_dir.glob("*")
        if file.name.casefold().endswith(".csv")
        and not file.name.casefold().startswith("split_")
    ]


def load_datasplit_indices(split_data) -> Tuple[npt.NDArray, npt.NDArray]:
    train_indices = split_data["split"] == "train"
    valid_indices = split_data["split"] == "valid"

    return train_indices.values, valid_indices.values


def validate_datasplit_data(
    landmark_data: pd.DataFrame, split_data: pd.DataFrame
) -> None:
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
