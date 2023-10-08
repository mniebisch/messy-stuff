from numpy import typing as npt

__all__ = ["compute_knuckle_direction"]


def compute_knuckle_direction(hand: npt.NDArray) -> tuple[float, float, float]:
    if hand.shape != (21, 3):
        raise ValueError("Incorrect landmark shape.")

    index_knuckle = hand[5]
    pinky_knuckle = hand[17]

    knuckle_direction = pinky_knuckle - index_knuckle
    return tuple(knuckle_direction)
