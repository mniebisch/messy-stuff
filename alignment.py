import numpy as np
from numpy import typing as npt


def shift_wrist_to_origin(hand: npt.NDArray) -> npt.NDArray:
    if hand.shape != (21, 3):
        raise ValueError("Incorrect landmark shape.")
    return hand - hand[[0]]


if __name__ == "__main__":
    dummy_hand = np.arange(21 * 3).reshape((21, 3))
    centered_dummy_hand = shift_wrist_to_origin(dummy_hand)

    print("Done")
