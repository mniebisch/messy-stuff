import itertools
from dataclasses import fields
import functools
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from numpy import typing as npt

from fmp.datasets.fingerspelling5 import metrics, utils


def compute_metric_combinations(
    hand: npt.NDArray,
    func_list: List[Callable],
    wrapper: Callable,
    argument_collection: Tuple[List[Any], ...],
) -> Dict[str, float]:
    # argument_collection must match order of input arguments (TODO make robust later)
    func_list = [wrapper(func) for func in func_list]
    iteration_collection = (func_list, *argument_collection)
    metric_results = [
        func(hand, *args) for func, *args in itertools.product(*iteration_collection)
    ]
    return functools.reduce(lambda x, y: {**x, **y}, metric_results)


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
    location_metrics = compute_metric_combinations(
        hand=hand,
        func_list=location_functions,
        wrapper=metrics.location_wrapper,
        argument_collection=(parts,),
    )

    # extract space (plane) metrics
    space_functions = [
        metrics.compute_hand_plane_area,
        metrics.compute_hand_plane_perimeter,
    ]
    space_metrics = compute_metric_combinations(
        hand=hand,
        func_list=space_functions,
        wrapper=metrics.space_wrapper,
        argument_collection=(planes, parts),
    )

    # extract distance metrics
    distance_functions = [metrics.compute_distances_mean, metrics.compute_distances_std]
    distance_metrics = compute_metric_combinations(
        hand=hand,
        func_list=distance_functions,
        wrapper=metrics.distance_wrapper,
        argument_collection=(axes, parts),
    )

    # extract angle metrics
    angle_functions = [metrics.compute_palm_angle, metrics.compute_knuckle_angle]
    angle_metrics = compute_metric_combinations(
        hand=hand,
        func_list=angle_functions,
        wrapper=metrics.angle_wrapper,
        argument_collection=(planes,),
    )

    print(
        len(angle_metrics),
        len(distance_metrics),
        len(space_metrics),
        len(location_metrics),
    )
