import numpy as np
import pytest

from fmp.datasets.fingerspelling5 import metrics


def test_compute_hand_extend():
    hand = np.zeros((21, 3))
    hand[0, 1] = -1
    hand[-1, 1] = 1
    hand[7, 2] = 4
    expected = np.array([0, 2, 4])
    output = metrics.compute_hand_extend(hand)
    np.testing.assert_array_equal(expected, output)


def test_compute_hand_mean():
    hand = np.zeros((21, 3))
    hand[:, 1] = 2
    hand[0, 2] = 21
    expected = np.array([0.0, 2.0, 1.0])
    output = metrics.compute_hand_mean(hand)
    np.testing.assert_array_equal(expected, output)


def test_compute_hand_std():
    hand = np.zeros((21, 3))
    hand[:, 1] = np.arange(21)
    expected = np.array([0, np.std(np.arange(21)), 0])
    output = metrics.compute_hand_std(hand)
    np.testing.assert_array_equal(expected, output)


def test_shape_check():
    bad_hand = np.zeros((4, 3))
    with pytest.raises(ValueError, match="Incorrect hand landmark shape."):
        metrics.compute_hand_std(bad_hand)
