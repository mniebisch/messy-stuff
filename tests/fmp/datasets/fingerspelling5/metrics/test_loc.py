import numpy as np
import pytest

from fmp.datasets.fingerspelling5.metrics import loc


def test_compute_hand_extend():
    hand = np.zeros((21, 3))
    hand[0, 1] = -1
    hand[-1, 1] = 1
    hand[7, 2] = 4
    expected = np.array([0, 2, 4])
    output = loc.compute_hand_extend(hand)
    np.testing.assert_array_equal(expected, output)


def test_compute_hand_mean():
    hand = np.zeros((21, 3))
    hand[:, 1] = 2
    hand[0, 2] = 21
    expected = np.array([0.0, 2.0, 1.0])
    output = loc.compute_hand_mean(hand)
    np.testing.assert_array_equal(expected, output)


def test_compute_hand_std():
    hand = np.zeros((21, 3))
    hand[:, 1] = np.arange(21)
    expected = np.array([0, np.std(np.arange(21)), 0])
    output = loc.compute_hand_std(hand)
    np.testing.assert_array_equal(expected, output)


def test_location_wrap():
    wrapped_func = loc.location_wrapper(loc.compute_hand_extend)
    hand = np.zeros((21, 3))
    part = "thumb"
    hand[1, 0] = 1
    hand[1, 1] = 2
    hand[1, 2] = 3
    expected = {
        "loc_thumb_x_extend": 1.0,
        "loc_thumb_y_extend": 2.0,
        "loc_thumb_z_extend": 3.0,
    }
    output = wrapped_func(hand, part)
    assert output == pytest.approx(expected)
