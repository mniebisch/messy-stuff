import collections

import numpy as np
from numpy import typing as npt

__all__ = ["compute_knuckle_direction", "compute_palm_direction"]

AngleSummary = collections.namedtuple("AngleSummary", "xy yz xz")


def compute_knuckle_direction(hand: npt.NDArray) -> tuple[float, float, float]:
    if hand.shape != (21, 3):
        raise ValueError("Incorrect landmark shape.")

    index_knuckle = hand[5]
    pinky_knuckle = hand[17]

    knuckle_direction = pinky_knuckle - index_knuckle
    return tuple(knuckle_direction)


def compute_palm_direction(hand: npt.NDArray) -> tuple[float, float, float]:
    if hand.shape != (21, 3):
        raise ValueError("Incorrect landmark shape.")

    wrist = hand[0]
    index_knuckle = hand[5]
    pinky_knuckle = hand[17]

    wrist_index_direction = index_knuckle - wrist
    wrist_pinky_direction = pinky_knuckle - wrist

    palm_direction = np.cross(wrist_index_direction, wrist_pinky_direction)
    return tuple(palm_direction)


def describe_angles(v1: npt.NDArray, v2: npt.NDArray) -> AngleSummary:
    if v1.shape != (3,) or v2.shape != (3,):
        raise ValueError("Vectors with incorrect shape.")

    xy_ind = [0, 1]
    xz_ind = [0, 2]
    yz_ind = [1, 2]

    xy_angle = angle_between(v1[xy_ind], v2[xy_ind])
    xz_angle = angle_between(v1[xz_ind], v2[xz_ind])
    yz_angle = angle_between(v1[yz_ind], v2[yz_ind])
    return AngleSummary(xy_angle, yz_angle, xz_angle)


def unit_vector(vector: npt.NDArray) -> npt.NDArray:
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1: npt.NDArray, v2: npt.NDArray) -> float:
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    Source: https://stackoverflow.com/a/13849249

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
