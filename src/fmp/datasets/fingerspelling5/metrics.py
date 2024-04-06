import inspect
from functools import wraps

import numpy as np
from numpy import typing as npt
from scipy import spatial
from scipy.spatial import distance

from . import utils

__all__ = [
    "angle_wrapper",
    "compute_hand_extend",
    "compute_distances_mean",
    "compute_distances_std",
    "compute_hand_mean",
    "compute_hand_std",
    "compute_hand_plane_area",
    "compute_hand_plane_perimeter",
    "compute_palm_direction",
    "compute_palm_angle",
    "distance_wrapper",
    "location_wrapper",
    "space_wrapper",
]


def check_hand_landmark_shape(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        hand = bound.arguments["hand"]
        num_nodes = utils.mediapipe_hand_landmarks.num_nodes
        spatial_dims = len(utils.mediapipe_hand_landmarks.spatial_coords)
        if hand.shape != (num_nodes, spatial_dims):
            raise ValueError("Incorrect hand landmark shape.")
        return func(*args, **kwargs)

    return wrapper


def location_wrapper(func):
    # TODO register allowed functions and verify on use?
    @wraps(func)
    def wrap_values(*args, **kwargs):
        signature = inspect.signature(func)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        part = bound.arguments["part"]
        metric_type = func.__name__.split("_")[-1]
        values = func(*args, **kwargs)
        return {
            f"{part}_{dim}_{metric_type}": val
            for val, dim in zip(values, utils.mediapipe_hand_landmarks.spatial_coords)
        }

    return wrap_values


def space_wrapper(func):
    # TODO register allowed functions and verify on use?
    def wrap_values(*args, **kwargs):
        signature = inspect.signature(func)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        part = bound.arguments["part"]
        metric_type = func.__name__.split("_")[-1]
        plane = bound.arguments["plane"]
        plane = "".join(plane)

        value = func(*args, **kwargs)
        return {f"{part}_{plane}_{metric_type}": value}

    return wrap_values


def distance_wrapper(func):
    # TODO register allowed functions and verify on use?
    def wrap_values(*args, **kwargs):
        signature = inspect.signature(func)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        part = bound.arguments["part"]
        metric_type = func.__name__.split("_")[-1]
        axis = bound.arguments["axis"]

        value = func(*args, **kwargs)
        return {f"{part}_{axis}_{metric_type}": value}

    return wrap_values


def angle_wrapper(func):
    # TODO register allowed functions and verify on use?
    def wrap_values(*args, **kwargs):
        signature = inspect.signature(func)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        plane = bound.arguments["plane"]
        plane = "".join(plane)

        direction_part = func.__name__.split("_")[-2]

        value = func(*args, **kwargs)
        return {f"{direction_part}_{plane}_angle": value}

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


def _compute_hand_convex_hull(
    hand: npt.NDArray, plane: tuple[str, str], part: str
) -> spatial.ConvexHull:
    part_indices = getattr(utils.mediapipe_hand_landmarks.parts, part)
    plane_ind = (
        utils.mediapipe_hand_landmarks.spatial_coords.index(plane[0]),
        utils.mediapipe_hand_landmarks.spatial_coords.index(plane[1]),
    )
    hand_part = hand[np.ix_(part_indices, plane_ind)]
    return spatial.ConvexHull(hand_part)


@check_hand_landmark_shape
def compute_hand_plane_perimeter(
    hand: npt.NDArray, plane: tuple[str, str], part: str = "all"
) -> float:
    convex_hull = _compute_hand_convex_hull(hand, plane, part)
    return convex_hull.area


@check_hand_landmark_shape
def compute_hand_plane_area(
    hand: npt.NDArray, plane: tuple[str, str], part: str = "all"
) -> float:
    convex_hull = _compute_hand_convex_hull(hand, plane, part)
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


def _compute_distances(hand: npt.NDArray, axis: str, part: str) -> npt.NDArray:
    plane_ind = [utils.mediapipe_hand_landmarks.spatial_coords.index(ax) for ax in axis]
    part_indices = getattr(utils.mediapipe_hand_landmarks.parts, part)
    hand_part = hand[np.ix_(part_indices, plane_ind)]
    dist = distance.cdist(hand_part, hand_part, metric="euclidean")
    assert dist.shape[0] == dist.shape[1], "dist has to be a square array!"
    rows, cols = np.triu_indices_from(dist, k=1)
    return dist[rows, cols]


@check_hand_landmark_shape
def compute_distances_mean(hand: npt.NDArray, axis: str, part: str = "all") -> float:
    # valid axis ("xyz", "xy", "yz", "xz", "x", "y", "z")
    dist_flat = _compute_distances(hand, axis, part)
    return np.mean(dist_flat)


@check_hand_landmark_shape
def compute_distances_std(hand: npt.NDArray, axis: str, part: str = "all") -> float:
    # valid axis ("xyz", "xy", "yz", "xz", "x", "y", "z")
    dist_flat = _compute_distances(hand, axis, part)
    return np.std(dist_flat)


@check_hand_landmark_shape
def compute_palm_angle(hand: npt.NDArray, plane: tuple[str, str]) -> float:
    palm_direction = compute_palm_direction(hand)
    return _compute_angle(palm_direction, plane)


def _compute_angle(direction: npt.NDArray, plane: tuple[str, str]) -> float:
    plane_ind = (
        utils.mediapipe_hand_landmarks.spatial_coords.index(plane[0]),
        utils.mediapipe_hand_landmarks.spatial_coords.index(plane[1]),
    )

    direction_projection = direction[np.ix_(plane_ind)]
    basis_projected_plane = np.array([1, 0])

    return angle_between(direction_projection, basis_projected_plane)


@check_hand_landmark_shape
def compute_knuckle_angle(hand: npt.NDArray, plane: tuple[str, str]) -> float:
    knuckle_direction = compute_knuckle_direction(hand)
    return _compute_angle(knuckle_direction, plane)


@check_hand_landmark_shape
def compute_palm_direction(hand: npt.NDArray) -> npt.NDArray:

    wrist = hand[utils.mediapipe_hand_landmarks.nodes.wrist]
    index_knuckle = hand[utils.mediapipe_hand_landmarks.nodes.index_mcp]
    pinky_knuckle = hand[utils.mediapipe_hand_landmarks.nodes.pinky_mcp]

    # right hand rule 'a' vector
    wrist_index_direction = index_knuckle - wrist
    # right hasnd rule 'b' vector
    wrist_pinky_direction = pinky_knuckle - wrist

    # considering right hand rule
    # and given assumption that right hand was recorded
    # cross product direction is towards the camera if the inner side of the hand
    # points towards the camera too
    palm_direction = np.cross(wrist_index_direction, wrist_pinky_direction)
    return palm_direction


@check_hand_landmark_shape
def compute_knuckle_direction(hand: npt.NDArray) -> npt.NDArray:
    """
    Knuckle direction as vector going from pinky knuckle to index knuckle.
    """
    index_knuckle = hand[utils.mediapipe_hand_landmarks.nodes.index_mcp]
    pinky_knuckle = hand[utils.mediapipe_hand_landmarks.nodes.pinky_mcp]

    return index_knuckle - pinky_knuckle


def unit_vector(vector: npt.NDArray) -> npt.NDArray:
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1: npt.NDArray, v2: npt.NDArray) -> float:
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    Source: https://stackoverflow.com/a/13849249

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))