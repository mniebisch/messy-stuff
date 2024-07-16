from typing import Tuple

import numpy as np
from numpy import typing as npt
from scipy import spatial
import torch
from torch_geometric import data as pyg_data
from torch_geometric import transforms as pyg_transforms

from fmp.datasets.fingerspelling5 import utils
from fmp.datasets.fingerspelling5.features import cdiff
from fmp.datasets.fingerspelling5.metrics import angle

__all__ = ["NodeCDiff", "RotateHand"]


class NodeCDiff(pyg_transforms.BaseTransform):
    def __call__(self, data: pyg_data.Data) -> pyg_data.Data:
        if data.pos is None:
            raise ValueError("Invalid 'pos' input.")

        if utils.is_any_invalid_attribute_set(data):
            raise ValueError(
                "Transform will invalidate previous node structure. "
                "No attribute besides 'pos' is allowed to be set."
            )

        data.pos = cdiff.cdiff_combination(data.pos)
        return data


class RotateHand(pyg_transforms.BaseTransform):
    def __init__(
        self,
        palm_normal: Tuple[float, float, float],
        knuckle_direction: Tuple[float, float, float],
    ) -> None:

        self.rotation_target = np.stack(
            [
                np.array(palm_normal),
                np.array(knuckle_direction),
            ]
        )

    def __call__(self, data: pyg_data.Data) -> pyg_data.Data:
        if data.pos is None:
            raise ValueError("Invalid 'pos' input.")

        dtype = data.pos.dtype

        hand = data.pos.numpy()
        hand_palm_normal = angle.compute_palm_direction(hand)
        knuckle_direction = angle.compute_knuckle_direction(hand)
        hand_directions = np.stack(([hand_palm_normal, knuckle_direction]))
        rotation_matrix = compute_rotation_matrix(
            vectors_aligned=self.rotation_target, vectors_misaligned=hand_directions
        )

        rotated_hand = np.dot(hand, rotation_matrix)
        data.pos = torch.from_numpy(rotated_hand)
        data.pos = data.pos.to(dtype)

        return data


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
