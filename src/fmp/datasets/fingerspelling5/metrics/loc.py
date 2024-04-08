import inspect
from functools import wraps

import numpy as np
from numpy import typing as npt

from . import utils as metric_utils
from .. import utils as fs5_utils


__all__ = [
    "compute_hand_extend",
    "compute_hand_mean",
    "compute_hand_std",
    "location_wrapper",
]


@metric_utils.check_hand_landmark_shape
def compute_hand_extend(hand: npt.NDArray, part: str = "all") -> npt.NDArray:
    part_indices = getattr(fs5_utils.mediapipe_hand_landmarks.parts, part)
    hand_part = hand[part_indices]
    min_vals = np.min(hand_part, axis=0)
    max_vals = np.max(hand_part, axis=0)
    return np.abs(max_vals - min_vals)


@metric_utils.check_hand_landmark_shape
def compute_hand_mean(hand: npt.NDArray, part: str = "all") -> npt.NDArray:
    part_indices = getattr(fs5_utils.mediapipe_hand_landmarks.parts, part)
    hand_part = hand[part_indices]
    return np.mean(hand_part, axis=0)


@metric_utils.check_hand_landmark_shape
def compute_hand_std(hand: npt.NDArray, part: str = "all") -> npt.NDArray:
    part_indices = getattr(fs5_utils.mediapipe_hand_landmarks.parts, part)
    hand_part = hand[part_indices]
    return np.std(hand_part, axis=0)


def location_wrapper(func):
    # TODO register allowed functions and verify on use?
    @wraps(func)
    def wrap_values(*args, **kwargs):
        signature = inspect.signature(func)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        part = bound.arguments["part"]
        metric_type = func.__name__.split("_")[-1]
        values = func(*args, **kwargs)
        return {
            f"loc_{part}_{dim}_{metric_type}": val
            for val, dim in zip(
                values, fs5_utils.mediapipe_hand_landmarks.spatial_coords
            )
        }

    return wrap_values
