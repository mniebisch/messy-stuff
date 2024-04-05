import inspect
from functools import wraps

import numpy as np
from numpy import typing as npt
from scipy import spatial
from scipy.spatial import distance

from . import utils

__all__ = [
    "compute_hand_extend",
    "compute_hand_mean",
    "compute_hand_std",
    "compute_hand_plane_area",
    "compute_hand_plane_perimeter",
]


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def check_hand_landmark_shape(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        hand = kwargs["hand"] if len(args) < 1 else args[0]
        num_nodes = utils.mediapipe_hand_landmarks.num_nodes
        spatial_dims = len(utils.mediapipe_hand_landmarks.spatial_coords)
        if hand.shape != (num_nodes, spatial_dims):
            raise ValueError("Incorrect hand landmark shape.")
        return func(*args, **kwargs)

    return wrapper


def location_wrapper(func):
    @wraps(func)
    def wrap_values(*args, **kwds):
        kwargs = get_default_args(func)
        kwargs.update(kwds)
        part = kwargs.get("part")
        part = args[1] if part is None else part
        metric_type = func.__name__.split("_")[-1]
        values = func(*args, **kwds)
        return {
            f"{part}_{dim}_{metric_type}": val
            for val, dim in zip(values, utils.mediapipe_hand_landmarks.spatial_coords)
        }

    return wrap_values


def space_wrapper(func):
    def wrap_values(*args, **kwds):
        kwargs = get_default_args(func)
        kwargs.update(kwds)
        part = kwargs.get("part")
        part = args[2] if part is None else part
        metric_type = func.__name__.split("_")[-1]
        plane = kwargs.get("plane")
        plane = args[1] if plane is None else plane
        plane = "".join(plane)

        value = func(*args, **kwargs)
        return {f"{part}_{plane}_{metric_type}": value}

    return wrap_values


@check_hand_landmark_shape
def compute_hand_extend(hand: npt.NDArray, part: str = "all") -> npt.NDArray:
    part_indices = getattr(utils.mediapipe_hand_landmarks.parts, part)
    hand_part = hand[part_indices]
    min_vals = np.min(hand_part, axis=0)
    max_vals = np.max(hand_part, axis=0)
    return np.abs(max_vals - min_vals)


@check_hand_landmark_shape
def compute_hand_mean(hand: npt.NDArray, part: str = "all") -> npt.NDArray:
    part_indices = getattr(utils.mediapipe_hand_landmarks.parts, part)
    hand_part = hand[part_indices]
    return np.mean(hand_part, axis=0)


@check_hand_landmark_shape
def compute_hand_std(hand: npt.NDArray, part: str = "all") -> npt.NDArray:
    part_indices = getattr(utils.mediapipe_hand_landmarks.parts, part)
    hand_part = hand[part_indices]
    return np.std(hand_part, axis=0)


@check_hand_landmark_shape
def compute_hand_plane_perimeter(
    hand: npt.NDArray, plane: tuple[str, str], part: str = "all"
) -> float:
    part_indices = getattr(utils.mediapipe_hand_landmarks.parts, part)
    plane_ind = (
        utils.mediapipe_hand_landmarks.spatial_coords.index(plane[0]),
        utils.mediapipe_hand_landmarks.spatial_coords.index(plane[1]),
    )
    hand_part = hand[np.ix_(part_indices, plane_ind)]
    convex_hull = spatial.ConvexHull(hand_part)
    return convex_hull.area


@check_hand_landmark_shape
def compute_hand_plane_area(
    hand: npt.NDArray, plane: tuple[str, str], part: str = "all"
) -> float:
    part_indices = getattr(utils.mediapipe_hand_landmarks.parts, part)
    plane_ind = (
        utils.mediapipe_hand_landmarks.spatial_coords.index(plane[0]),
        utils.mediapipe_hand_landmarks.spatial_coords.index(plane[1]),
    )
    hand_part = hand[np.ix_(part_indices, plane_ind)]
    convex_hull = spatial.ConvexHull(hand_part)
    vertices = convex_hull.points[convex_hull.vertices]
    return shoelace_formula(vertices)


def shoelace_formula(vertices: npt.NDArray) -> float:
    """
    Compute the area of a 2D convex polygon using the shoelace formula.

    The shoelace formula calculates the area of a polygon by summing the products
    of the coordinates of adjacent vertices and then dividing by 2.

    Requirements:
    1. Convexity: The polygon should be convex. For concave polygons, split them
       into multiple convex polygons for accurate results.
    2. Vertex Order: Vertices must be specified in either clockwise or
       counterclockwise order.
    3. Closed Polygon: Ensure that the first and last vertices of the array are
       the same to close the polygon.
    4. Non-Self-Intersecting: The polygon should not have any self-intersections.
    5. Non-Negative Area: The result will be positive if vertices are ordered
       counterclockwise, and negative if ordered clockwise. To ensure a positive
       result, order the vertices counterclockwise.
    6. No Duplicates: Vertices should not contain duplicate points.

    Parameters:
        vertices (npt.NDArray): Array of shape [N x 2] representing the vertices
            of the polygon.

    Returns:
        float: Area of the convex polygon.

    Example:
        >>> polygon_vertices = np.array([[0, 0], [4, 0], [4, 3], [1, 2]])
        >>> area = shoelace_formula(polygon_vertices)
        >>> print(f"The area of the polygon is {area}")
    """

    if vertices.shape[1] != 2:
        raise ValueError("Function only holds for 2D space.")

    if vertices.shape[0] < 3:
        raise ValueError("A polygon must have at least three vertices.")

    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
