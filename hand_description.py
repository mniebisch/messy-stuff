import collections

import numpy as np
from numpy import typing as npt
from scipy import spatial
from scipy.spatial import distance

__all__ = [
    "compute_extend",
    "compute_distance_adjacency",
    "compute_hand_mean",
    "compute_hand_std",
    "compute_knuckle_direction",
    "compute_palm_direction",
    "compute_plane_shape_stats",
]

AngleSummary = collections.namedtuple("AngleSummary", "xy yz xz")
PolygonStats = collections.namedtuple("PolygonStats", "area perimeter")
Planes = collections.namedtuple("Planes", "xy yz xz")


def compute_extend(hand: npt.NDArray) -> tuple[float, float, float]:
    if hand.shape != (21, 3):
        raise ValueError("Incorrect landmark shape.")

    min_vals = np.min(hand, axis=0)
    max_vals = np.max(hand, axis=0)
    extend = np.abs(max_vals - min_vals)
    return tuple(extend)


def compute_knuckle_direction(hand: npt.NDArray) -> tuple[float, float, float]:
    """
    Knuckle direction as vector going from pinky knuckle to index knuckle.
    """
    if hand.shape != (21, 3):
        raise ValueError("Incorrect landmark shape.")

    index_knuckle = hand[5]
    pinky_knuckle = hand[17]

    knuckle_direction = index_knuckle - pinky_knuckle
    return tuple(knuckle_direction)


def compute_palm_direction(hand: npt.NDArray) -> tuple[float, float, float]:
    if hand.shape != (21, 3):
        raise ValueError("Incorrect landmark shape.")

    wrist = hand[0]
    index_knuckle = hand[5]
    pinky_knuckle = hand[17]

    # right hand rule 'a' vector
    wrist_index_direction = index_knuckle - wrist
    # right hasnd rule 'b' vector
    wrist_pinky_direction = pinky_knuckle - wrist

    # considering right hand rule
    # and given assumption that right hand was recorded
    # cross product direction is towards the camera if the inner side of the hand
    # points towards the camera too
    palm_direction = np.cross(wrist_index_direction, wrist_pinky_direction)
    return tuple(palm_direction)


def describe_angles(v1: npt.NDArray, v2: npt.NDArray) -> AngleSummary:
    if v1.shape != (3,) or v2.shape != (3,):
        raise ValueError("Vectors with incorrect shape.")

    xy_ind = [0, 1]
    xz_ind = [0, 2]
    yz_ind = [1, 2]

    xy_angle = angle_between(v1[xy_ind], v2[xy_ind])
    xz_angle = angle_between(v1[xz_ind], v2[xz_ind])
    yz_angle = angle_between(v1[yz_ind], v2[yz_ind])
    return AngleSummary(xy_angle, yz_angle, xz_angle)


def compute_distance_adjacency(hand: npt.NDArray, dim: str = "all") -> npt.NDArray:
    dims = {"all": [0, 1, 2], "x": [0], "y": [1], "z": [2]}
    dim_indices = dims[dim]
    if hand.shape != (21, 3):
        raise ValueError("Incorrect landmark shape.")
    return distance.cdist(hand[:, dim_indices], hand[dim_indices], metric="euclidean")


def compute_hand_mean(hand: npt.NDArray, part: str = "all") -> npt.NDArray:
    if hand.shape != (21, 3):
        raise ValueError("Incorrect landmark shape.")

    parts = {
        "all": list(range(21)),
        "thumb": [1, 2, 3, 4],
        "index_finger": [5, 6, 7, 8],
        "middle_finger": [9, 10, 11, 12],
        "ring_finger": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }
    indices = parts[part]
    return np.mean(hand[indices], axis=0)


def compute_hand_std(hand: npt.NDArray, part: str = "all") -> npt.NDArray:
    if hand.shape != (21, 3):
        raise ValueError("Incorrect landmark shape.")

    parts = {
        "all": list(range(21)),
        "thumb": [1, 2, 3, 4],
        "index_finger": [5, 6, 7, 8],
        "middle_finger": [9, 10, 11, 12],
        "ring_finger": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }
    indices = parts[part]
    return np.std(hand[indices], axis=0)


def compute_polygon_stats(polygon: spatial.ConvexHull) -> PolygonStats:
    area = shoelace_formula(polygon.points[polygon.vertices])
    perimeter = polygon.area
    return PolygonStats(area, perimeter)


def compute_plane_shape_stats(hand: npt.NDArray) -> Planes[PolygonStats]:
    if hand.shape != (21, 3):
        raise ValueError("Incorrect landmark shape.")

    stats_xy = compute_polygon_stats(spatial.ConvexHull(hand[:, :2]))
    stats_yz = compute_polygon_stats(spatial.ConvexHull(hand[:, 1:]))
    stats_xz = compute_polygon_stats(spatial.ConvexHull(hand[:, [0, 2]]))
    return Planes(stats_xy, stats_yz, stats_xz)


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
