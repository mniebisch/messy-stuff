import numpy as np
import pytest

from fmp.datasets.fingerspelling5.metrics import space


def test_compute_hand_plane_area():
    hand = np.zeros((21, 3))
    square_points = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
        ]
    )
    hand[:4, [0, 1]] = square_points
    expected = 1.0
    output = space.compute_hand_plane_area(hand, ("x", "y"))
    assert output == pytest.approx(expected)


def test_compute_hand_plane_perimeter():
    hand = np.zeros((21, 3))
    square_points = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
        ]
    )
    hand[:4, [0, 1]] = square_points
    expected = 4.0
    output = space.compute_hand_plane_perimeter(hand, ("x", "y"))
    assert output == pytest.approx(expected)


def test_space_wrap():
    wrapped_func = space.space_wrapper(space.compute_hand_plane_area)
    hand = np.zeros((21, 3))
    part = "pinky"
    plane = ("x", "y")
    square_points = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
        ]
    )
    hand[17:21, [0, 1]] = square_points
    expected = {"space_pinky_xy_area": 1.0}
    output = wrapped_func(hand, plane, part)
    assert output == pytest.approx(expected)
