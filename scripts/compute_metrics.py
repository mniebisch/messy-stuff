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


# TODO create class and put setup into init
def compute_metrics(hand: npt.NDArray) -> Dict[str, float]:
    # Setup argument iterables
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

    # Setup functions
    # TODO crawl from to be created modules to not forget update?
    location_functions = [
        metrics.compute_hand_mean,
        metrics.compute_hand_std,
        metrics.compute_hand_extend,
    ]
    space_functions = [
        metrics.compute_hand_plane_area,
        metrics.compute_hand_plane_perimeter,
    ]
    distance_functions = [metrics.compute_distances_mean, metrics.compute_distances_std]
    angle_functions = [metrics.compute_palm_angle, metrics.compute_knuckle_angle]

    iterables: list[tuple[list[Callable], Callable, Tuple[List[Any], ...]]] = [
        (location_functions, metrics.location_wrapper, (parts,)),
        (space_functions, metrics.space_wrapper, (planes, parts)),
        (distance_functions, metrics.distance_wrapper, (axes, parts)),
        (angle_functions, metrics.angle_wrapper, (planes,)),
    ]
    # compute results
    results = [
        compute_metric_combinations(
            hand=hand,
            func_list=func_list,
            wrapper=wrapper,
            argument_collection=argument_collection,
        )
        for func_list, wrapper, argument_collection in iterables
    ]
    return functools.reduce(lambda x, y: {**x, **y}, results)


if __name__ == "__main__":
    hand = np.random.rand(21, 3)
    metrics_output = compute_metrics(hand=hand)
