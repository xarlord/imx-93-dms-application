"""Geometric utilities: distance, angle, polygon test, EAR, MAR."""
from typing import Any

import numpy as np
from numpy.typing import NDArray


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Compute Euclidean distance using NumPy.

    Args:
        p1: First point as (x, y).
        p2: Second point as (x, y).

    Returns:
        The straight-line distance.
    """
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def midpoint(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
    """Return the midpoint of the segment between two points.

    Args:
        p1: First point as (x, y).
        p2: Second point as (x, y).

    Returns:
        The midpoint as (x, y).
    """
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def compute_ear(eye_landmarks: list[tuple[float, float]]) -> float:
    """Compute the Eye Aspect Ratio (EAR) from 6 eye landmarks.

    Landmarks order: [outer_corner, upper1, upper2, inner_corner, lower1, lower2].

    Args:
        eye_landmarks: Six (x, y) landmarks around one eye.

    Returns:
        EAR value, typically ~0.30 (open) to ~0.05 (closed).
    """
    v1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    h = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_mar(mouth_landmarks: list[tuple[float, float]]) -> float:
    """Compute the Mouth Aspect Ratio (MAR) from 8 mouth landmarks.

    Landmarks order: [left_corner, upper1, upper2, upper3, right_corner,
        lower1, lower2, lower3].

    Args:
        mouth_landmarks: Eight (x, y) landmarks around the mouth.

    Returns:
        MAR value, typically ~0.3 (closed) to ~1.0+ (yawning).
    """
    v1 = euclidean_distance(mouth_landmarks[1], mouth_landmarks[7])
    v2 = euclidean_distance(mouth_landmarks[2], mouth_landmarks[6])
    v3 = euclidean_distance(mouth_landmarks[3], mouth_landmarks[5])
    h = euclidean_distance(mouth_landmarks[0], mouth_landmarks[4])
    if h < 1e-6:
        return 0.0
    return (v1 + v2 + v3) / (3.0 * h)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Test whether a point lies inside a polygon using the ray-casting algorithm.

    Args:
        point: The (x, y) point to test.
        polygon: Ordered list of (x, y) vertices defining the polygon.

    Returns:
        True if *point* is inside *polygon*.
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
        j = i
    return inside


def classify_zone(
    point: tuple[float, float],
    zones: list[tuple[int, str, str, list[tuple[float, float]]]],
    zone_order: list[int],
) -> tuple[int, str]:
    """Find which zone a point belongs to.

    Zones are checked in ascending area order so that smaller zones take
    priority over larger enclosing zones.

    Args:
        point: The (x, y) gaze point to classify.
        zones: List of (zone_id, name, type, polygon) tuples.
        zone_order: Zone IDs sorted by polygon area ascending (small first).

    Returns:
        A (zone_id, zone_name) tuple. Returns (15, 'UNKNOWN') when the
        point falls outside every zone.
    """
    for zone_id in zone_order:
        for z_id, z_name, z_type, polygon in zones:
            if z_id == zone_id and point_in_polygon(point, polygon):
                return (z_id, z_name)
    return (15, 'UNKNOWN')


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    """Compute the area of a simple polygon via the shoelace formula.

    Args:
        polygon: Ordered list of (x, y) vertices.

    Returns:
        The absolute area of the polygon.
    """
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def compute_head_pose(
    landmarks: list[tuple[float, float]],
    face_bbox: list[float],
) -> tuple[float, float]:
    """Estimate head yaw and pitch from nose-tip offset relative to face centre.

    This is a 2D approximation: yaw is derived from the horizontal offset of
    the nose tip from the mid-eye point, normalised by face width; pitch is
    derived from the vertical offset normalised by face height.

    .. warning::
        This is a 2D approximation using nose-tip horizontal offset from
        mid-eye as yaw proxy. For head turns >= 30 degrees the nose moves
        nonlinearly and the bounding box shifts, producing degenerate
        readings. The gaze blend in GazeEstimator weights this at only
        40% to mitigate, but accuracy degrades at large head angles.
        A proper 3D PnP solver (e.g. solvePnP with a 3D face model)
        should replace this for production use.

    Args:
        landmarks: 25-point landmarks where index 0 is the nose tip.
        face_bbox: Bounding box as [x1, y1, x2, y2].

    Returns:
        A (yaw_deg, pitch_deg) tuple. Returns (0.0, 0.0) when the
        bounding box is degenerate.
    """
    x1, y1, x2, y2 = face_bbox
    face_w = x2 - x1
    face_h = y2 - y1
    if face_w < 1e-6 or face_h < 1e-6:
        return (0.0, 0.0)

    face_cx = (x1 + x2) / 2.0
    face_cy = (y1 + y2) / 2.0

    nose = landmarks[0]

    left_eye_cx = sum(landmarks[i][0] for i in range(5, 11)) / 6.0
    left_eye_cy = sum(landmarks[i][1] for i in range(5, 11)) / 6.0
    right_eye_cx = sum(landmarks[i][0] for i in range(11, 17)) / 6.0
    right_eye_cy = sum(landmarks[i][1] for i in range(11, 17)) / 6.0

    mid_eye_x = (left_eye_cx + right_eye_cx) / 2.0
    mid_eye_y = (left_eye_cy + right_eye_cy) / 2.0

    yaw = (nose[0] - mid_eye_x) / face_w * 60.0
    pitch = (nose[1] - mid_eye_y) / face_h * 45.0

    return (yaw, pitch)
