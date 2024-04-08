import inspect
from functools import wraps

from .. import utils as fs5_utils

__all__ = ["check_hand_landmark_shape"]


def check_hand_landmark_shape(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        hand = bound.arguments["hand"]
        num_nodes = fs5_utils.mediapipe_hand_landmarks.num_nodes
        spatial_dims = len(fs5_utils.mediapipe_hand_landmarks.spatial_coords)
        if hand.shape != (num_nodes, spatial_dims):
            raise ValueError("Incorrect hand landmark shape.")
        return func(*args, **kwargs)

    return wrapper
