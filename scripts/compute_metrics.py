import itertools
from dataclasses import fields
import functools

import numpy as np

from fmp.datasets.fingerspelling5 import metrics, utils


if __name__ == "__main__":
    hand = np.random.rand(21, 3)
    parts = [field.name for field in fields(utils.mediapipe_hand_landmarks.parts)]
    planes = list(
        itertools.combinations(utils.mediapipe_hand_landmarks.spatial_coords, 2)
    )
    axes = [
        itertools.combinations(utils.mediapipe_hand_landmarks.spatial_coords, n)
        for n in range(1, len(utils.mediapipe_hand_landmarks.spatial_coords) + 1)
    ]
    axes = [list(axis) for axis in axes]
    axes = list(itertools.chain(*axes))
    axes = ["".join(axis) for axis in axes]
    # extract location metrics
    location_functions = [
        metrics.compute_hand_mean,
        metrics.compute_hand_std,
        metrics.compute_hand_extend,
    ]
    location_functions = [metrics.location_wrapper(func) for func in location_functions]
    location_metrics = [
        func(hand, part) for part, func in itertools.product(parts, location_functions)
    ]
    location_metrics = functools.reduce(lambda x, y: {**x, **y}, location_metrics)

    # extract space (plane) metrics
    space_functions = [
        metrics.compute_hand_plane_area,
        metrics.compute_hand_plane_perimeter,
    ]
    space_functions = [metrics.space_wrapper(func) for func in space_functions]
    space_metrics = [
        func(hand, plane, part)
        for plane, part, func in itertools.product(planes, parts, space_functions)
    ]
    space_metrics = functools.reduce(lambda x, y: {**x, **y}, space_metrics)

    # extract distance metrics
    distance_functions = [metrics.compute_distances_mean, metrics.compute_distances_std]
    distance_functions = [metrics.distance_wrapper(func) for func in distance_functions]
    distance_metrics = [
        func(hand, axis, part)
        for axis, part, func in itertools.product(axes, parts, distance_functions)
    ]
    distance_metrics = functools.reduce(lambda x, y: {**x, **y}, distance_metrics)

    # extract angle metrics
    angle_functions = [metrics.compute_palm_angle, metrics.compute_knuckle_angle]
    angle_functions = [metrics.angle_wrapper(func) for func in angle_functions]
    angle_metrics = [
        func(hand, plane) for plane, func in itertools.product(planes, angle_functions)
    ]
    angle_metrics = functools.reduce(lambda x, y: {**x, **y}, angle_metrics)

    print(
        len(angle_metrics),
        len(distance_metrics),
        len(space_metrics),
        len(location_metrics),
    )
