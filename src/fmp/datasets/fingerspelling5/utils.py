from typing import Any, List, Union

import numpy as np
import torch
from numpy import typing as npt
from torch_geometric import data as pyg_data

__all__ = [
    "generate_hand_landmark_columns",
    "OneHotLabel",
    "ReshapeToTriple",
    "Flatten",
    "PyGDataWrapper",
    "PyGDataUnwrapper",
    "NDArrayToTensor",
]

# Use dataclass instead?
MEDIAPIPE_HAND_LANDMARKS = {
    "num_nodes": 21,
    "spatial_coords": ("x", "y", "z"),
}


# MediaPipe property!
# Checkout pragmatic programmer
def generate_hand_landmark_columns() -> List[str]:
    num_nodes = MEDIAPIPE_HAND_LANDMARKS["num_nodes"]
    spatial_coords = MEDIAPIPE_HAND_LANDMARKS["spatial_coords"]

    return [
        f"{spatial_coord}_hand_{node_ind}"
        for node_ind in range(num_nodes)
        for spatial_coord in spatial_coords
    ]


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
        return graph.pos


class NDArrayToTensor(object):
    def __call__(self, data: npt.NDArray) -> torch.Tensor:
        return torch.from_numpy(data)
