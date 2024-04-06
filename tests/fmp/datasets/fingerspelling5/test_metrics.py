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
    output = metrics.compute_hand_plane_area(hand, ("x", "y"))
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
    output = metrics.compute_hand_plane_perimeter(hand, ("x", "y"))
    assert output == pytest.approx(expected)


def test_location_wrap():
    wrapped_func = metrics.location_wrapper(metrics.compute_hand_extend)
    hand = np.zeros((21, 3))
    part = "thumb"
    hand[1, 0] = 1
    hand[1, 1] = 2
    hand[1, 2] = 3
    expected = {"thumb_x_extend": 1.0, "thumb_y_extend": 2.0, "thumb_z_extend": 3.0}
    output = wrapped_func(hand, part)
    assert output == pytest.approx(expected)


def test_plane_wrap():
    wrapped_func = metrics.space_wrapper(metrics.compute_hand_plane_area)
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
    expected = {"pinky_xy_area": 1.0}
    output = wrapped_func(hand, plane, part)
    assert output == pytest.approx(expected)


def test_compute_distances_mean_1d():
    hand = np.zeros((21, 3))
    part = "thumb"
    axis = "x"
    hand[1, 0] = 1
    expected = 3 / 6
    output = metrics.compute_distances_mean(hand, axis, part)
    assert output == pytest.approx(expected)


def test_compute_distances_mean_2d():
    hand = np.zeros((21, 3))
    part = "index_finger"
    axis = "yz"
    hand[5, 1] = 1
    expected = 3 / 6
    output = metrics.compute_distances_mean(hand, axis, part)
    assert output == pytest.approx(expected)


def test_compute_distances_mean_3d():
    hand = np.zeros((21, 3))
    axis = "xyz"
    hand[0, 0] = 1
    expected = 20 / 210
    output = metrics.compute_distances_mean(hand, axis)
    assert output == pytest.approx(expected)


def test_distance_wrap():
    func = metrics.distance_wrapper(metrics.compute_distances_mean)
    hand = np.zeros((21, 3))
    part = "index_finger"
    axis = "yz"
    hand[5, 1] = 1
    expected = {"index_finger_yz_mean": 3 / 6}
    output = func(hand, axis, part)
    assert output == pytest.approx(expected)


def test_compute_palm_angle():
    hand = np.zeros((21, 3))
    hand[5] = [1, 1, 0]
    hand[17] = [1, -1, 0]
    plane = ("x", "z")
    expected = np.pi / 2
    output = metrics.compute_palm_angle(hand, plane)
    assert output == pytest.approx(expected)


def test_compute_knuckle_angle():
    hand = np.zeros((21, 3))
    hand[5] = [1, 1, 0]
    plane = ("x", "y")
    expected = np.pi / 4
    output = metrics.compute_knuckle_angle(hand, plane)
    assert output == pytest.approx(expected)


def test_angle_wrap():
    func = metrics.angle_wrapper(metrics.compute_knuckle_angle)
    hand = np.zeros((21, 3))
    hand[5] = [1, -1, 0]
    plane = ("y", "z")
    expected = {"knuckle_yz_angle": np.pi}
    output = func(hand, plane)
    assert output == pytest.approx(expected)
