import numpy as np
import pytest

from fmp.datasets.fingerspelling5.metrics import angle


def test_compute_palm_angle():
    hand = np.zeros((21, 3))
    hand[5] = [1, 1, 0]
    hand[17] = [1, -1, 0]
    plane = ("x", "z")
    expected = np.pi / 2
    output = angle.compute_palm_angle(hand, plane)
    assert output == pytest.approx(expected)


def test_compute_knuckle_angle():
    hand = np.zeros((21, 3))
    hand[5] = [1, 1, 0]
    plane = ("x", "y")
    expected = np.pi / 4
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
