import numpy as np
import pytest

from fmp.datasets.fingerspelling5.metrics import utils


def test_shape_check():
    bad_hand = np.zeros((4, 3))

    def foo(hand):
        return hand

    with pytest.raises(ValueError, match="Incorrect hand landmark shape."):
        test_foo = utils.check_hand_landmark_shape(foo)
        test_foo(bad_hand)
