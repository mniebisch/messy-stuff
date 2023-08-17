import functools
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import torchdata
from numpy import typing as npt


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


def load_fingerspelling5(
    hand_landmark_data: pd.DataFrame,
    batch_size: int = 64,
    drop_last: bool = True,
    filter_nan: bool = False,
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

    return datapipe


if __name__ == "__main__":
    data_path = pathlib.Path(__file__).parent / "data"
    train_csv = data_path / "fingerspelling5_singlehands.csv"

    train_data = pd.read_csv(train_csv)

    fu = load_fingerspelling5(train_data, batch_size=64, filter_nan=True)
    for batch, labels in fu:
        print("oi")
