import functools
import pathlib
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric.data as pyg_data
import torchdata
from numpy import typing as npt
from torchvision import transforms

import pipeline_pyg_augmentation as geometric_pipe_utils


def generate_hand_landmark_columns() -> List[str]:
    num_nodes = 21
    spatial_coords = ("x", "y", "z")

    return [
        f"{spatial_coord}_hand_{node_ind}"
        for node_ind in range(num_nodes)
        for spatial_coord in spatial_coords
    ]


def cast_float32(inputs: Tuple[npt.NDArray, str, str]) -> Tuple[npt.NDArray, str, str]:
    landmarks, person, letter = inputs
    return landmarks.astype(np.float32), person, letter


def fill_nan(inputs: Tuple[npt.NDArray, str, str]) -> Tuple[npt.NDArray, str, str]:
    landmarks, person, letter = inputs
    return np.nan_to_num(landmarks), person, letter


def letter_to_number(letter: str) -> int:
    if letter == "j" or letter == "z":
        raise ValueError("Letters 'j' and 'z' are not allowed.")

    if "a" <= letter <= "i":
        return ord(letter) - ord("a")
    elif "k" <= letter <= "y":
        return ord(letter) - ord("a") - 1
    else:
        raise ValueError("Invalid input: lowercase letters only.")


def map_label(inputs: Tuple[npt.NDArray, str, str]) -> Tuple[npt.NDArray, int]:
    landmarks, _, letter = inputs
    return landmarks, letter_to_number(letter)


def create_one_hot(
    inputs: Tuple[npt.NDArray, int], num_classes: int
) -> Tuple[npt.NDArray, npt.NDArray]:
    landmarks, label = inputs
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[label] = 1
    return landmarks, one_hot


def nan_filter(inputs: Tuple[npt.NDArray, int]) -> bool:
    coords, _, _ = inputs
    return bool(np.logical_not(np.any(np.isnan(coords))))


def landmarks_to_geom_datapoint(
    inputs: Tuple[npt.NDArray, npt.NDArray]
) -> pyg_data.Data:
    landmarks, one_hot = inputs
    geom_landmarks = geometric_pipe_utils.create_geom_datapoint(
        torch.from_numpy(landmarks)
    )
    # TODO not sure if adding one hot like this is good style
    geom_landmarks.one_hot = one_hot
    return geom_landmarks


def geom_datapoint_to_landmark(
    inputs: pyg_data.Data,
) -> Tuple[npt.NDArray, npt.NDArray]:
    one_hot = inputs.one_hot
    landmarks = geometric_pipe_utils.unwrap_pyg_datapoint(inputs)
    # TODO is .numpy() slow?
    return landmarks.numpy(), one_hot


class Scale(object):
    def __call__(self, batch):
        batch = batch - batch.mean(dim=-2, keepdim=True)
        scale = (1 / batch.abs().max()) * 0.999999
        batch = batch * scale
        return batch


class ReshapeToTriple(object):
    def __call__(self, batch):
        batch_size = batch.shape[0]
        batch = batch.reshape((batch_size, -1, 3))
        return batch


class FlattenTriple(object):
    def __call__(self, batch):
        batch_size = batch.shape[0]
        batch = batch.reshape((batch_size, -1))
        return batch


def load_fingerspelling5(
    hand_landmark_data: pd.DataFrame,
    batch_size: int = 64,
    drop_last: bool = True,
    filter_nan: bool = False,
    transform: Optional[object] = None,
):
    num_letters = ord("z") - ord("a") + 1 - 2

    one_hot = functools.partial(create_one_hot, num_classes=num_letters)

    landmark_cols = generate_hand_landmark_columns()
    person_col = "person"
    letter_col = "letter"
    landmark_data = hand_landmark_data[landmark_cols].to_numpy()
    person_data = hand_landmark_data[person_col].to_numpy()
    letter_data = hand_landmark_data[letter_col].to_numpy()

    datapipe = torchdata.datapipes.iter.IterableWrapper(
        zip(landmark_data, person_data, letter_data)
    )

    datapipe = datapipe.map(cast_float32)
    if filter_nan:
        datapipe = datapipe.filter(nan_filter)
    else:
        datapipe = datapipe.map(fill_nan)
    datapipe = datapipe.map(map_label)
    datapipe = datapipe.map(one_hot)

    datapipe = datapipe.shuffle(buffer_size=100000)
    datapipe = datapipe.batch(batch_size=batch_size, drop_last=drop_last)
    datapipe = datapipe.collate()
    if transform is not None:
        datapipe = datapipe.map(lambda x: (transform(x[0]), x[1]))

    return datapipe


if __name__ == "__main__":
    data_path = pathlib.Path(__file__).parent / "data"
    train_csv = data_path / "fingerspelling5_singlehands.csv"

    train_data = pd.read_csv(train_csv)

    trans = transforms.Compose([ReshapeToTriple(), Scale(), FlattenTriple()])

    fu = load_fingerspelling5(
        train_data, batch_size=64, filter_nan=True, transform=trans
    )
    for batch, labels in fu:
        print("oi")
