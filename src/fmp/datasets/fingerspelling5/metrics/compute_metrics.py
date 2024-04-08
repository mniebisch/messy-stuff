import itertools
from dataclasses import fields
import functools
from typing import Any, Callable, Dict, List, Tuple

from numpy import typing as npt

from fmp.datasets.fingerspelling5 import metrics, utils


__all__ = ["FingerspellingMetrics"]


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


class FingerspellingMetrics:
    def __init__(self):
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
            metrics.loc.compute_hand_mean,
            metrics.loc.compute_hand_std,
            metrics.loc.compute_hand_extend,
        ]
        space_functions = [
            metrics.space.compute_hand_plane_area,
            metrics.space.compute_hand_plane_perimeter,
        ]
        distance_functions = [
            metrics.dist.compute_distances_mean,
            metrics.dist.compute_distances_std,
        ]
        angle_functions = [
            metrics.angle.compute_palm_angle,
            metrics.angle.compute_knuckle_angle,
        ]

        self.iterables = [
            (location_functions, metrics.loc.location_wrapper, (parts,)),
            (space_functions, metrics.space.space_wrapper, (planes, parts)),
            (distance_functions, metrics.dist.distance_wrapper, (axes, parts)),
            (angle_functions, metrics.angle.angle_wrapper, (planes,)),
        ]

    def __call__(self, hand: npt.NDArray) -> Dict[str, float]:
        results = [
            compute_metric_combinations(
                hand=hand,
                func_list=func_list,
                wrapper=wrapper,
                argument_collection=argument_collection,
            )
            for func_list, wrapper, argument_collection in self.iterables
        ]

        return functools.reduce(lambda x, y: {**x, **y}, results)
