from dataclasses import dataclass
import pathlib
import string
from typing import List, Union

import numpy as np
from numpy import typing as npt
import pandas as pd
import torch
from torch_geometric import data as pyg_data

__all__ = [
    "generate_hand_landmark_columns",
    "OneHotLabel",
    "ReshapeToTriple",
    "Flatten",
    "PyGDataWrapper",
    "PyGDataUnwrapper",
    "NDArrayToTensor",
    "mediapipe_hand_landmarks",
    "fingerspelling5",
    "read_csv",
    "is_any_invalid_attribute_set",
]

# Use dataclass instead?
MEDIAPIPE_HAND_LANDMARKS = {
    "num_nodes": 21,
    "spatial_coords": ("x", "y", "z"),
}


@dataclass(frozen=True)
class HandParts:
    all: list[int]
    thumb: list[int]
    index_finger: list[int]
    middle_finger: list[int]
    ring_finger: list[int]
    pinky: list[int]
    palm: list[int]
    wrist: list[int]


@dataclass(frozen=True)
class HandNodes:
    # mcp => kinda knuckle
    wrist: int = 0
    index_mcp: int = 5
    middle_mcp: int = 9
    ring_mcp: int = 13
    pinky_mcp: int = 17


@dataclass(frozen=True)
class MediaPipeHandLandmarks:
    parts: HandParts
    nodes: HandNodes = HandNodes()
    num_nodes: int = 21
    spatial_coords: tuple[str, str, str] = ("x", "y", "z")


@dataclass(frozen=True)
class Fingerspelling5:
    letters: List[str]


mediapipe_hand_landmarks = MediaPipeHandLandmarks(
    parts=HandParts(
        all=list(range(21)),
        thumb=[1, 2, 3, 4],
        index_finger=[5, 6, 7, 8],
        middle_finger=[9, 10, 11, 12],
        ring_finger=[13, 14, 15, 16],
        pinky=[17, 18, 19, 20],
        palm=[0, 5, 9, 13, 17],
        wrist=[0],
    )
)

fingerspelling5 = Fingerspelling5(
    letters=[letter for letter in string.ascii_lowercase if letter not in ("j", "z")]
)


# MediaPipe property!
# Checkout pragmatic programmer
def generate_hand_landmark_columns() -> List[str]:
    num_nodes = mediapipe_hand_landmarks.num_nodes
    spatial_coords = mediapipe_hand_landmarks.spatial_coords

    return [
        f"{spatial_coord}_hand_{node_ind}"
        for node_ind in range(num_nodes)
        for spatial_coord in spatial_coords
    ]


def read_csv(csv_file: Union[str, pathlib.Path], filter_nans: bool) -> pd.DataFrame:
    landmark_data = pd.read_csv(csv_file)
    if filter_nans:
        landmark_data = landmark_data.loc[~landmark_data.isnull().any(axis=1)]
        landmark_data = landmark_data.reset_index()

    return landmark_data


def is_any_invalid_attribute_set(graph: pyg_data.Data) -> bool:
    return any(
        [value is not None and key != "pos" for key, value in graph.to_dict().items()]
    )


# could be general purpose
class OneHotLabel(object):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, label: int) -> npt.NDArray:
        one_hot = np.zeros(self.num_classes, dtype=np.bool_)
        one_hot[label] = True
        return one_hot


class ReshapeToTriple(object):
    def __init__(self):
        self.num_spatial_coords = len(MEDIAPIPE_HAND_LANDMARKS["spatial_coords"])
        self.num_nodes = MEDIAPIPE_HAND_LANDMARKS["num_nodes"]

    def __call__(self, flattened_landmarks: Union[npt.NDArray, torch.Tensor]):
        return flattened_landmarks.reshape((self.num_nodes, self.num_spatial_coords))


class Flatten(object):
    def __call__(self, input_data: Union[npt.NDArray, torch.Tensor]):
        # .reshape((1, -1)) ??
        return input_data.reshape((-1))


class PyGDataWrapper(object):
    def __call__(self, pos_data: torch.Tensor) -> pyg_data.Data:
        return pyg_data.Data(pos=pos_data)


class PyGDataUnwrapper(object):
    def __call__(self, graph: pyg_data.Data) -> torch.Tensor:
        if is_any_invalid_attribute_set(graph):
            raise ValueError(
                "Transform will invalidate previous node structure. "
                "No attribute besides 'pos' is allowed to be set."
                "Please check the pipeline."
            )
        return graph.pos


class NDArrayToTensor(object):
    def __call__(self, data: npt.NDArray) -> torch.Tensor:
        return torch.from_numpy(data)
