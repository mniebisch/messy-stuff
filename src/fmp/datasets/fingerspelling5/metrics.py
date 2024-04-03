import collections

import numpy as np
from numpy import typing as npt
from scipy import spatial
from scipy.spatial import distance

from . import utils

__all__ = [
    "compute_hand_extend",
    "compute_hand_mean",
    "compute_hand_std",
]


def check_hand_landmark_shape(func):
    def wrapper(*args, **kwargs):
        hand = kwargs["hand"] if len(args) < 1 else args[0]
        num_nodes = utils.mediapipe_hand_landmarks.num_nodes
        spatial_dims = len(utils.mediapipe_hand_landmarks.spatial_coords)
        if hand.shape != (num_nodes, spatial_dims):
            raise ValueError("Incorrect hand landmark shape.")
        return func(*args, **kwargs)

    return wrapper


def xyz_part_wrapper(func):
    def wrap_values(*args, **kwargs):
        part = kwargs["part"] if len(args) < 2 else args[1]
        metric_type = func.__name__.split("_")[-1]
        values = func(*args, **kwargs)
        return {
            f"{part}_{dim}_{metric_type}": val
            for val, dim in zip(values, utils.mediapipe_hand_landmarks.spatial_coords)
        }

    return wrap_values


@check_hand_landmark_shape
def compute_hand_extend(hand: npt.NDArray, part: str = "all") -> npt.NDArray:
    part_indices = getattr(utils.mediapipe_hand_landmarks.parts, part)
    hand_part = hand[part_indices]
    min_vals = np.min(hand_part, axis=0)
    max_vals = np.max(hand_part, axis=0)
    return np.abs(max_vals - min_vals)


@check_hand_landmark_shape
def compute_hand_mean(hand: npt.NDArray, part: str = "all") -> npt.NDArray:
    part_indices = getattr(utils.mediapipe_hand_landmarks.parts, part)
    hand_part = hand[part_indices]
    return np.mean(hand_part, axis=0)


@check_hand_landmark_shape
def compute_hand_std(hand: npt.NDArray, part: str = "all") -> npt.NDArray:
    part_indices = getattr(utils.mediapipe_hand_landmarks.parts, part)
    hand_part = hand[part_indices]
    return np.std(hand_part, axis=0)
