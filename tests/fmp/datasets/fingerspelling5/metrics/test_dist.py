import numpy as np
import pytest

from fmp.datasets.fingerspelling5.metrics import dist


def test_compute_distances_mean_1d():
    hand = np.zeros((21, 3))
    part = "thumb"
    axis = "x"
    hand[1, 0] = 1
    expected = 3 / 6
    output = dist.compute_distances_mean(hand, axis, part)
    assert output == pytest.approx(expected)


def test_compute_distances_mean_2d():
    hand = np.zeros((21, 3))
    part = "index_finger"
    axis = "yz"
    hand[5, 1] = 1
    expected = 3 / 6
    output = dist.compute_distances_mean(hand, axis, part)
    assert output == pytest.approx(expected)


def test_compute_distances_mean_3d():
    hand = np.zeros((21, 3))
    axis = "xyz"
    hand[0, 0] = 1
    expected = 20 / 210
    output = dist.compute_distances_mean(hand, axis)
    assert output == pytest.approx(expected)


def test_distance_wrap():
    func = dist.distance_wrapper(dist.compute_distances_mean)
    hand = np.zeros((21, 3))
    part = "index_finger"
    axis = "yz"
    hand[5, 1] = 1
    expected = {"dist_index_finger_yz_mean": 3 / 6}
    output = func(hand, axis, part)
    assert output == pytest.approx(expected)
