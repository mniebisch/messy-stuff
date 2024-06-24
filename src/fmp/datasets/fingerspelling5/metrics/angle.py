import inspect
from functools import wraps

import numpy as np
from numpy import typing as npt

from . import utils as metric_utils
from .. import utils as fs5_utils


__all__ = ["angle_wrapper", "compute_knuckle_angle", "compute_palm_angle"]


@metric_utils.check_hand_landmark_shape
def compute_knuckle_angle(hand: npt.NDArray, plane: tuple[str, str]) -> float:
    knuckle_direction = compute_knuckle_direction(hand)
    return _compute_angle(knuckle_direction, plane)


@metric_utils.check_hand_landmark_shape
def compute_palm_angle(hand: npt.NDArray, plane: tuple[str, str]) -> float:
    palm_direction = compute_palm_direction(hand)
    return _compute_angle(palm_direction, plane)


def angle_wrapper(func):
    # TODO register allowed functions and verify on use?
    @wraps(func)
    def wrap_values(*args, **kwargs):
        signature = inspect.signature(func)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        plane = bound.arguments["plane"]
        plane = "".join(plane)

        direction_part = func.__name__.split("_")[-2]

        value = func(*args, **kwargs)
        return {f"angle_{direction_part}_{plane}_angle": value}

    return wrap_values


@metric_utils.check_hand_landmark_shape
def compute_palm_direction(hand: npt.NDArray) -> npt.NDArray:

    wrist = hand[fs5_utils.mediapipe_hand_landmarks.nodes.wrist]
    index_knuckle = hand[fs5_utils.mediapipe_hand_landmarks.nodes.index_mcp]
    pinky_knuckle = hand[fs5_utils.mediapipe_hand_landmarks.nodes.pinky_mcp]

    # right hand rule 'a' vector
    wrist_index_direction = index_knuckle - wrist
    # right hasnd rule 'b' vector
    wrist_pinky_direction = pinky_knuckle - wrist

    # considering right hand rule
    # and given assumption that right hand was recorded
    # cross product direction is towards the camera if the inner side of the hand
    # points towards the camera too
    palm_direction = np.cross(wrist_index_direction, wrist_pinky_direction)
    return palm_direction


@metric_utils.check_hand_landmark_shape
def compute_knuckle_direction(hand: npt.NDArray) -> npt.NDArray:
    """
    Knuckle direction as vector going from pinky knuckle to index knuckle.
    """
    index_knuckle = hand[fs5_utils.mediapipe_hand_landmarks.nodes.index_mcp]
    pinky_knuckle = hand[fs5_utils.mediapipe_hand_landmarks.nodes.pinky_mcp]

    return index_knuckle - pinky_knuckle


def angle_between(v1: npt.NDArray, v2: npt.NDArray) -> float:
    angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))

    return angle


def _compute_angle(direction: npt.NDArray, plane: tuple[str, str]) -> float:
    plane_ind = (
        fs5_utils.mediapipe_hand_landmarks.spatial_coords.index(plane[0]),
        fs5_utils.mediapipe_hand_landmarks.spatial_coords.index(plane[1]),
    )
    direction_projection = direction[np.ix_(plane_ind)]
    basis_projected_plane = np.array([1, 0])

    return angle_between(basis_projected_plane, direction_projection)
