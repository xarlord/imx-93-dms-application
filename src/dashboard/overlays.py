"""Dashboard overlays: landmarks, gaze ray, zone indicators on camera feed."""
try:
    import cv2
    HAS_CV2: bool = True
except ImportError:
    HAS_CV2 = False

import math

import numpy as np
from numpy.typing import NDArray


# Landmark color groups
LANDMARK_COLORS: dict[str, tuple[int, int, int]] = {
    'nose': (0, 255, 255),      # yellow
    'brow': (255, 200, 0),       # cyan
    'left_eye': (255, 0, 0),     # blue
    'right_eye': (0, 100, 255),  # orange
    'mouth': (200, 200, 200),    # gray
}

LANDMARK_GROUPS: dict[str, list[int]] = {
    'nose': [0],
    'brow': [1, 2, 3, 4],
    'left_eye': list(range(5, 11)),
    'right_eye': list(range(11, 17)),
    'mouth': list(range(17, 25)),
}


def draw_landmarks(frame: NDArray[np.uint8], landmarks: list[tuple[float, float]] | None, scale_x: float = 1.0, scale_y: float = 1.0) -> None:
    """Draw 25 facial landmarks grouped by region with colored dots.

    Args:
        frame: BGR numpy array to draw on (modified in place).
        landmarks: List of 25 ``(x, y)`` tuples in original frame coords,
            or ``None`` to skip.
        scale_x: Horizontal scaling factor from frame to display coords.
        scale_y: Vertical scaling factor from frame to display coords.
    """
    if not HAS_CV2 or landmarks is None:
        return

    for group_name, indices in LANDMARK_GROUPS.items():
        color: tuple[int, int, int] = LANDMARK_COLORS[group_name]
        for idx in indices:
            if idx < len(landmarks):
                x: int = int(landmarks[idx][0] * scale_x)
                y: int = int(landmarks[idx][1] * scale_y)
                cv2.circle(frame, (x, y), 3, color, -1)

    # Connect eye landmarks with lines
    for eye_range in [(5, 11), (11, 17)]:
        pts: list[tuple[int, int]] = []
        for idx in range(eye_range[0], eye_range[1]):
            if idx < len(landmarks):
                pts.append((int(landmarks[idx][0] * scale_x),
                           int(landmarks[idx][1] * scale_y)))
        if len(pts) >= 4:
            cv2.polylines(frame, [np.array(pts)], True, (0, 255, 0), 1)


def draw_iris(frame: NDArray[np.uint8], left_iris: tuple[float, float, float] | None, right_iris: tuple[float, float, float] | None, scale_x: float = 1.0, scale_y: float = 1.0) -> None:
    """Draw iris circles on camera frame for both eyes.

    Args:
        frame: BGR numpy array to draw on (modified in place).
        left_iris: ``(cx, cy, r)`` for the left eye, or ``None``.
        right_iris: ``(cx, cy, r)`` for the right eye, or ``None``.
        scale_x: Horizontal scaling factor from frame to display coords.
        scale_y: Vertical scaling factor from frame to display coords.
    """
    if not HAS_CV2:
        return

    for iris in [left_iris, right_iris]:
        if iris is not None:
            cx: int = int(iris[0] * scale_x)
            cy: int = int(iris[1] * scale_y)
            r: int = max(1, int(iris[2] * min(scale_x, scale_y)))
            cv2.circle(frame, (cx, cy), r, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)


def draw_gaze_ray(frame: NDArray[np.uint8], nose_tip: tuple[float, float] | None, final_yaw: float, final_pitch: float,
                  length: int = 150, scale_x: float = 1.0, scale_y: float = 1.0) -> None:
    """Draw a gaze direction arrow from the nose tip.

    Args:
        frame: BGR numpy array to draw on (modified in place).
        nose_tip: ``(x, y)`` landmark, or ``None`` to skip.
        final_yaw: Stabilized yaw angle in degrees.
        final_pitch: Stabilized pitch angle in degrees.
        length: Arrow length in pixels.
        scale_x: Horizontal scaling factor.
        scale_y: Vertical scaling factor.
    """
    if not HAS_CV2 or nose_tip is None:
        return

    x0: int = int(nose_tip[0] * scale_x)
    y0: int = int(nose_tip[1] * scale_y)
    # Gaze endpoint
    dx: int = int(math.sin(math.radians(final_yaw)) * length)
    dy: int = int(math.sin(math.radians(final_pitch)) * length)
    x1: int = x0 + dx
    y1: int = y0 + dy

    cv2.arrowedLine(frame, (x0, y0), (x1, y1), (0, 0, 255), 2, tipLength=0.15)


def draw_head_bbox(frame: NDArray[np.uint8], bbox: list[float] | None, confidence: float = 0.0, scale_x: float = 1.0, scale_y: float = 1.0) -> None:
    """Draw head bounding box with optional confidence label.

    Args:
        frame: BGR numpy array to draw on (modified in place).
        bbox: ``[x1, y1, x2, y2]`` in original frame coords, or ``None``.
        confidence: Detection confidence score (0 to hide label).
        scale_x: Horizontal scaling factor.
        scale_y: Vertical scaling factor.
    """
    if not HAS_CV2 or bbox is None:
        return

    x1: int = int(bbox[0] * scale_x)
    y1: int = int(bbox[1] * scale_y)
    x2: int = int(bbox[2] * scale_x)
    y2: int = int(bbox[3] * scale_y)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if confidence > 0:
        cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
