import inspect
from functools import wraps

import numpy as np
from numpy import typing as npt
from scipy.spatial import distance

from . import utils as metric_utils
from .. import utils as fs5_utils


__all__ = ["compute_distances_mean", "compute_distances_std", "distance_wrapper"]


@metric_utils.check_hand_landmark_shape
def compute_distances_mean(hand: npt.NDArray, axis: str, part: str = "all") -> float:
    # valid axis ("xyz", "xy", "yz", "xz", "x", "y", "z")
    dist_flat = _compute_distances(hand, axis, part)
    return np.mean(dist_flat)


@metric_utils.check_hand_landmark_shape
def compute_distances_std(hand: npt.NDArray, axis: str, part: str = "all") -> float:
    # valid axis ("xyz", "xy", "yz", "xz", "x", "y", "z")
    dist_flat = _compute_distances(hand, axis, part)
    return np.std(dist_flat)


def distance_wrapper(func):
    # TODO register allowed functions and verify on use?
    @wraps(func)
    def wrap_values(*args, **kwargs):
        signature = inspect.signature(func)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        part = bound.arguments["part"]
        metric_type = func.__name__.split("_")[-1]
        axis = bound.arguments["axis"]

        value = func(*args, **kwargs)
        return {f"dist_{part}_{axis}_{metric_type}": value}

    return wrap_values


def _compute_distances(hand: npt.NDArray, axis: str, part: str) -> npt.NDArray:
    plane_ind = [
        fs5_utils.mediapipe_hand_landmarks.spatial_coords.index(ax) for ax in axis
    ]
    part_indices = getattr(fs5_utils.mediapipe_hand_landmarks.parts, part)
    hand_part = hand[np.ix_(part_indices, plane_ind)]
    dist = distance.cdist(hand_part, hand_part, metric="euclidean")
    assert dist.shape[0] == dist.shape[1], "dist has to be a square array!"
    rows, cols = np.triu_indices_from(dist, k=1)
    return dist[rows, cols]
