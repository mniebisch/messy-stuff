import string
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from . import utils

__all__ = ["Fingerspelling5Landmark"]

FINGERSPELLING5 = {
    "letters": [letter for letter in string.ascii_lowercase if letter not in ("j", "z")]
}


class Fingerspelling5Landmark(Dataset):
    def __init__(
        self,
        hand_landmark_data: pd.DataFrame,
        transforms=None,
        filter_nans: bool = False,
    ):
        # transforms: expected to be PyTorch Geometric transforms

        # hand_landmark_data are is a dataframe where we already applied
        # mediapipe to fingerspelling5 dataset

        # fingerspelling5 'properties'
        self.letters = FINGERSPELLING5["letters"]
        self.num_letters = len(self.letters)

        if filter_nans:
            landmark_data = hand_landmark_data.loc[
                ~hand_landmark_data.isnull().any(axis=1)
            ]
            landmark_data = landmark_data.reset_index()
        else:
            landmark_data = hand_landmark_data
        self._landmark_data = landmark_data

        # landmark data 'properties'
        self.num_features = utils.MEDIAPIPE_HAND_LANDMARKS["num_nodes"]
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
