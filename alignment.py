import numpy as np
from numpy import typing as npt
from scipy import spatial


def shift_wrist_to_origin(hand: npt.NDArray) -> npt.NDArray:
    if hand.shape != (21, 3):
        raise ValueError("Incorrect landmark shape.")
    return hand - hand[[0]]


def compute_rotation_matrix(vectors_aligned, vectors_misaligned) -> npt.NDArray:
    """

    Example:
        >>> vectors_aligned = np.array([[2, -1, 0]])
        >>> vectors_misaligned = np.array([[1, 1, 1]])
        >>> rotation_matrix = compute_rotation_matrix(vectors_algined, vectors_misaligned)
        >>> vectors_corrected = np.dot(vectors_misaligned, rotation_matrix)
        >>> # vectors_corrected and vectors_aligned have same direction now
    """

    # initialize rotation object
    identity = np.diag(np.full(3, 1))
    rotation = spatial.transform.Rotation.from_matrix(identity)

    # compute alignment
    rotation, _ = rotation.align_vectors(vectors_misaligned, vectors_aligned)

    return rotation.as_matrix()


if __name__ == "__main__":
    dummy_hand = np.arange(21 * 3).reshape((21, 3))
    centered_dummy_hand = shift_wrist_to_origin(dummy_hand)

    print("Done")
