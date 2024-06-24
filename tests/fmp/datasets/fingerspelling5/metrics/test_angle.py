from typing import List

import numpy as np
import pytest

from fmp.datasets.fingerspelling5.metrics import angle


@pytest.mark.parametrize(
    "index_knuckle,pinky_knuckle,expected_angle",
    [
        ([1, -1, 0], [1, 1, 0], np.pi / 2),
        ([1, 1, 0], [1, -1, 0], -np.pi / 2),
    ],
)
def test_compute_palm_angle(
    index_knuckle: List, pinky_knuckle: List, expected_angle: float
):
    hand = np.zeros((21, 3))
    hand[5] = index_knuckle
    hand[17] = pinky_knuckle
    plane = ("x", "z")
    expected = expected_angle
    output = angle.compute_palm_angle(hand, plane)
    assert output == pytest.approx(expected)


@pytest.mark.parametrize(
    "vector_input,expected_angle",
    [
        ([1, 0, 0], 0),
        ([1, 1, 0], np.pi / 4),
        ([0, 1, 0], np.pi / 2),
        ([-1, 0, 0], np.pi),
        ([0, -1, 0], -np.pi / 2),
        ([-1, -1, 0], -3 * np.pi / 4),
        ([2, 2, 0], np.pi / 4),
    ],
)
def test_compute_knuckle_angle(vector_input: List, expected_angle: float):
    hand = np.zeros((21, 3))
    hand[5] = vector_input
    plane = ("x", "y")
    expected = expected_angle
    output = angle.compute_knuckle_angle(hand, plane)
    assert output == pytest.approx(expected)


def test_angle_wrap():
    func = angle.angle_wrapper(angle.compute_knuckle_angle)
    hand = np.zeros((21, 3))
    hand[5] = [1, -1, 0]
    plane = ("y", "z")
    expected = {"angle_knuckle_yz_angle": np.pi}
    output = func(hand, plane)
    assert output == pytest.approx(expected)
