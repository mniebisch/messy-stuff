import pathlib
from typing import Optional, Tuple

import kornia as K
import numpy as np
import pandas as pd
import torch
from numpy import typing as npt
from torch.utils.data import Dataset
from torchvision.transforms import v2

from . import utils

__all__ = [
    "Fingerspelling5Image",
    "Fingerspelling5ImageInmemory",
    "Fingerspelling5Landmark",
    "setup_transforms",
]


class Fingerspelling5Landmark(Dataset):
    def __init__(
        self,
        hand_landmark_data: pd.DataFrame,
        transforms=None,
        split: Optional[str] = None,
    ):
        self.split = split
        # transforms: expected to be PyTorch Geometric transforms

        # hand_landmark_data are is a dataframe where we already applied
        # mediapipe to fingerspelling5 dataset

        # fingerspelling5 'properties'
        self.letters = utils.fingerspelling5.letters
        self.num_letters = len(self.letters)

        self._landmark_data = hand_landmark_data

        # landmark data 'properties'
        self.num_features = utils.mediapipe_hand_landmarks.num_nodes * 3
        self._landmark_cols = utils.generate_hand_landmark_columns()
        self._person_col = "person"
        self._letter_col = "letter"

        expected_columns = {*self._landmark_cols, self._person_col, self._letter_col}
        available_columns = set(self._landmark_data)

        if not expected_columns.issubset(available_columns):
            raise ValueError

        self._label_transforms = self._setup_label_transforms()
        self._transforms = self._setup_transforms(transforms)

    def __len__(self) -> int:
        return len(self._landmark_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self._landmark_data.iloc[idx]
        person = sample.loc[self._person_col]
        letter = sample.loc[self._letter_col]
        landmarks = sample.loc[self._landmark_cols].values
        landmarks = landmarks.astype(np.float32)

        label = self._label_transforms(self.letters.index(letter))
        data = self._transforms(landmarks)

        return data, label

    def _setup_label_transforms(self):
        return v2.Compose(
            [
                utils.OneHotLabel(self.num_letters),
                utils.NDArrayToTensor(),
                v2.ToDtype(torch.float32),
            ]
        )

    def _setup_landmark_pre_transforms(self):
        return v2.Compose(
            [
                utils.NDArrayToTensor(),
                v2.ToDtype(torch.float32),
                utils.ReshapeToTriple(),
                utils.PyGDataWrapper(),
            ]
        )

    def _setup_landmark_post_transforms(self):
        return v2.Compose([utils.PyGDataUnwrapper(), utils.Flatten()])

    def _setup_transforms(self, transforms):
        pre_transforms = self._setup_landmark_pre_transforms()
        post_transforms = self._setup_landmark_post_transforms()
        trans = (
            [pre_transforms, post_transforms]
            if transforms is None
            else [pre_transforms, transforms, post_transforms]
        )
        return v2.Compose(trans)


class Fingerspelling5Image(Dataset):
    def __init__(
        self,
        file_data: pd.DataFrame,
        dataset_path: pathlib.Path,
        tv_transforms: Optional[v2.Transform] = None,
        kornia_transforms: Optional[K.augmentation.AugmentationSequential] = None,
        split: Optional[str] = None,
    ) -> None:
        self.split = split

        # fingerspelling5 'properties'
        self.letters = utils.fingerspelling5.letters
        self.num_letters = len(self.letters)

        self._label_transforms = self._setup_label_transforms()

        self.tv_transforms = tv_transforms
        self.kornia_transforms = kornia_transforms

        self.file_data = file_data
        self.dataset_path = dataset_path

    def __len__(self) -> int:
        return len(self.file_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.file_data.iloc[idx]
        image_file = sample["img_file"]
        label = sample["letter"]

        if self.tv_transforms is not None:
            desired_type = K.io.ImageLoadType.RGB8
        else:
            desired_type = K.io.ImageLoadType.RGB32

        image = K.io.load_image(
            self.dataset_path / image_file,
            desired_type=desired_type,
            device="cpu",
        )

        # it is expected that tv transforms contain
        # v2.ToImage()
        # at the beginning
        # and
        # v2.ToDtype(torch.float32, scale=True)
        # v2.ToPureTensor()
        # at the end
        if self.tv_transforms is not None:
            image = self.tv_transforms(image)

        if self.kornia_transforms is not None:
            image = self.kornia_transforms(image)

        label = self._label_transforms(self.letters.index(label))

        return image, label

    def _setup_label_transforms(self):
        return v2.Compose(
            [
                utils.OneHotLabel(self.num_letters),
                utils.NDArrayToTensor(),
                v2.ToDtype(torch.float32),
            ]
        )


class Fingerspelling5ImageInmemory(Dataset):
    def __init__(
        self,
        image_data: torch.Tensor,
        # file_data: pd.DataFrame,
        # dataset_path: pathlib.Path,
        labels: npt.NDArray,
        tv_transforms: Optional[v2.Transform] = None,
        kornia_transforms: Optional[K.augmentation.AugmentationSequential] = None,
        split: Optional[str] = None,
    ) -> None:
        self.split = split

        # fingerspelling5 'properties'
        self.letters = utils.fingerspelling5.letters
        self.num_letters = len(self.letters)

        self._label_transforms = self._setup_label_transforms()

        self.tv_transforms = tv_transforms
        self.kornia_transforms = kornia_transforms

        self.image_data = image_data
        self.labels = labels

    def __len__(self) -> int:
        return self.image_data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label = self.labels[idx]
        image = self.image_data[idx]

        # it is expected that tv transforms contain
        # v2.ToImage()
        # at the beginning
        # and
        # v2.ToDtype(torch.float32, scale=True)
        # v2.ToPureTensor()
        # at the end
        if self.tv_transforms is not None:
            image = self.tv_transforms(image)

        if self.kornia_transforms is not None:
            image = self.kornia_transforms(image)

        label = self._label_transforms(self.letters.index(label))

        return image, label

    def _setup_label_transforms(self):
        return v2.Compose(
            [
                utils.OneHotLabel(self.num_letters),
                utils.NDArrayToTensor(),
                v2.ToDtype(torch.float32),
            ]
        )


def setup_transforms(transforms: Optional[v2.Transform] = None) -> v2.Compose:
    conversion = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    if transforms is None:
        transforms = conversion
    else:
        transforms = v2.Compose([transforms, conversion])

    return transforms
