import numpy as np
from numpy import typing as npt
from scipy import spatial


def shift_wrist_to_origin(hand: npt.NDArray) -> npt.NDArray:
    if hand.shape != (21, 3):
        raise ValueError("Incorrect landmark shape.")
    return hand - hand[[0]]


def compute_rotation_matrix(
    vectors_aligned: npt.NDArray, vectors_misaligned: npt.NDArray
) -> npt.NDArray:
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

    origin = [0, 0, 0]
    point1 = [1, 1, 1]
    point2 = [2, -1, 0]

    vector1 = np.array(point1) - np.array(origin)
    vector2 = np.array(point2) - np.array(origin)

    vector1 = vector1.reshape((1, 3))
    vector2 = vector2.reshape((1, 3))

    rot_mat = compute_rotation_matrix(
        vectors_algined=vector2, vectors_misaligned=vector1
    )
    vector1_rotated = np.dot(vector1, rot_mat)

    print("Done")
