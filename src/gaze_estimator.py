"""Gaze estimation: iris-center ratio + head pose blend + zone projection + stabilization."""
from __future__ import annotations

from typing import Any

import math
import numpy as np
from numpy.typing import NDArray

from .utils.geometry import compute_head_pose


# Landmark indices for 25-point scheme
NOSE_TIP: int = 0
LEFT_EYE_START: int = 5
LEFT_EYE_END: int = 11
RIGHT_EYE_START: int = 11
RIGHT_EYE_END: int = 17
MOUTH_START: int = 17
MOUTH_END: int = 25


class GazeEstimator:
    """Geometric gaze estimation from iris positions and head pose.

    Pipeline:
        1. Compute iris-center ratio for each eye (horizontal + vertical)
        2. Blend iris gaze with head pose contribution (60/40 weighting)
        3. Apply calibration offsets
        4. Stabilize with adaptive EMA
        5. Project onto dashboard reference plane
        6. Classify into gaze zone via polygon test
    """

    def __init__(self, yaw_weight: float = 32.0, pitch_weight: float = 45.0,
                 ppd_yaw: float = 5.0, ppd_pitch: float = 5.0,
                 display_w: int = 1920, display_h: int = 1080,
                 alpha_fixation: float = 0.3, alpha_saccade: float = 0.8,
                 saccade_velocity: float = 15.0,
                 eye_weight: float = 0.6, head_weight: float = 0.4,
                 head_yaw_limit: float = 30.0, head_pitch_limit: float = 30.0) -> None:
        """Initialize the GazeEstimator with blending and stabilization params.

        Args:
            yaw_weight: Degrees per unit iris-ratio for yaw.
            pitch_weight: Degrees per unit iris-ratio for pitch.
            ppd_yaw: Pixels-per-degree for screen projection (yaw).
            ppd_pitch: Pixels-per-degree for screen projection (pitch).
            display_w: Display width in pixels.
            display_h: Display height in pixels.
            alpha_fixation: EMA smoothing alpha during fixations (low).
            alpha_saccade: EMA smoothing alpha during saccades (high).
            saccade_velocity: Angular velocity threshold to detect saccades.
            eye_weight: Weight for eye-based gaze (rest goes to head pose).
            head_weight: Weight for head-pose-based gaze.
            head_yaw_limit: Yaw angle beyond which eye-only gaze is used.
            head_pitch_limit: Pitch angle beyond which eye-only gaze is used.
        """
        self.yaw_weight: float = yaw_weight
        self.pitch_weight: float = pitch_weight
        self.ppd_yaw: float = ppd_yaw
        self.ppd_pitch: float = ppd_pitch
        self.display_w: int = display_w
        self.display_h: int = display_h
        self.alpha_fixation: float = alpha_fixation
        self.alpha_saccade: float = alpha_saccade
        self.saccade_velocity: float = saccade_velocity
        self.eye_weight: float = eye_weight
        self.head_weight: float = head_weight
        self.head_yaw_limit: float = head_yaw_limit
        self.head_pitch_limit: float = head_pitch_limit

        # Calibration offsets (set by gaze_calibration.py)
        self.yaw_offset: float = 0.0
        self.pitch_offset: float = 0.0

        # Stabilization state
        self.smooth_yaw: float = 0.0
        self.smooth_pitch: float = 0.0
        self.prev_raw_yaw: float = 0.0
        self.prev_raw_pitch: float = 0.0
        self.initialized: bool = False

        # Zone classifier (set externally via set_zone_loader)
        self._zone_loader: Any = None
        self._last_h_span: float = 30.0

    def set_zone_loader(self, zone_loader: Any) -> None:
        """Wire a ZoneLoader instance for gaze zone classification.

        Args:
            zone_loader: ``ZoneLoader`` providing ``classify((x, y))``.
        """
        self._zone_loader = zone_loader

    def set_calibration(self, yaw_offset: float, pitch_offset: float) -> None:
        """Set gaze calibration offsets subtracted from raw estimates.

        Args:
            yaw_offset: Yaw offset in degrees.
            pitch_offset: Pitch offset in degrees.
        """
        self.yaw_offset = yaw_offset
        self.pitch_offset = pitch_offset

    def compute_iris_ratio(self, eye_landmarks: list[tuple[float, float]],
                           iris_center: tuple[float, float]) -> tuple[float, float]:
        """Compute normalized iris position within eye bounds.

        Args:
            eye_landmarks: 6 points ``[outer, upper1, upper2, inner, lower1, lower2]``.
            iris_center: ``(x, y)`` in the same coordinate space.

        Returns:
            ``(horizontal_ratio, vertical_ratio)`` roughly in ``[-1, +1]``.
            Positive means screen-right / screen-down.
        """
        outer = eye_landmarks[0]
        inner = eye_landmarks[3]
        upper_pts = [eye_landmarks[1], eye_landmarks[2]]
        lower_pts = [eye_landmarks[4], eye_landmarks[5]]

        y_top = min(p[1] for p in upper_pts)
        y_bottom = max(p[1] for p in lower_pts)

        # Determine screen-left and screen-right corners
        x_left = min(outer[0], inner[0])
        x_right = max(outer[0], inner[0])

        # Horizontal: iris position from screen-left to screen-right
        h_span = x_right - x_left
        self._last_h_span: float = h_span
        if abs(h_span) < 1e-6:
            iris_x = 0.0
        else:
            iris_x = (iris_center[0] - x_left) / h_span
            iris_x = (iris_x - 0.5) * 2.0

        # Vertical: iris position between upper and lower lid
        v_span = y_bottom - y_top
        if abs(v_span) < 1e-6:
            iris_y = 0.0
        else:
            iris_y = (iris_center[1] - y_top) / v_span
            iris_y = (iris_y - 0.5) * 2.0

        return (iris_x, iris_y)

    def _compute_iris_angles(self, landmarks: list[tuple[float, float]],
                             left_iris: tuple[float, float, float],
                             right_iris: tuple[float, float, float]) -> tuple[float, float, float, float]:
        """Compute eye-based yaw/pitch from averaged iris ratios.

        Args:
            landmarks: 25 facial landmarks.
            left_iris: ``(cx, cy, r)`` for the left eye.
            right_iris: ``(cx, cy, r)`` for the right eye.

        Returns:
            ``(eye_yaw, eye_pitch, avg_iris_x, avg_iris_y)`` in degrees
            and ratio units respectively.
        """
        left_eye_lm = landmarks[LEFT_EYE_START:LEFT_EYE_END]
        right_eye_lm = landmarks[RIGHT_EYE_START:RIGHT_EYE_END]

        left_iris_x, left_iris_y = self.compute_iris_ratio(
            left_eye_lm, (left_iris[0], left_iris[1]))
        right_iris_x, right_iris_y = self.compute_iris_ratio(
            right_eye_lm, (right_iris[0], right_iris[1]))

        avg_iris_x = (left_iris_x + right_iris_x) / 2.0
        avg_iris_y = (left_iris_y + right_iris_y) / 2.0

        eye_yaw = avg_iris_x * self.yaw_weight
        eye_pitch = avg_iris_y * self.pitch_weight

        return eye_yaw, eye_pitch, avg_iris_x, avg_iris_y

    def _blend_with_head_pose(self, eye_yaw: float, eye_pitch: float,
                              face_bbox: list[float],
                              landmarks: list[tuple[float, float]]) -> tuple[float, float, float, float, float]:
        """Blend eye-based gaze with head pose using adaptive weights.

        Head weight tapers to zero as yaw/pitch approach their limits,
        gracefully falling back to eye-only gaze for large head turns.

        Args:
            eye_yaw: Eye-based yaw in degrees.
            eye_pitch: Eye-based pitch in degrees.
            face_bbox: ``[x1, y1, x2, y2]`` face bounding box.
            landmarks: 25 facial landmarks.

        Returns:
            ``(raw_yaw, raw_pitch, head_yaw, head_pitch, gaze_confidence)``.
        """
        head_yaw, head_pitch = compute_head_pose(landmarks, face_bbox)

        effective_head_weight = self.head_weight
        taper_start = 0.8
        for angle, limit in [(abs(head_yaw), self.head_yaw_limit),
                             (abs(head_pitch), self.head_pitch_limit)]:
            if angle > limit:
                effective_head_weight = 0.0
                break
            elif angle > taper_start * limit:
                taper = 1.0 - (angle - taper_start * limit) / (limit - taper_start * limit)
                effective_head_weight = min(effective_head_weight, self.head_weight * taper)
        effective_eye_weight = 1.0 - effective_head_weight
        raw_yaw = eye_yaw * effective_eye_weight + head_yaw * effective_head_weight
        raw_pitch = eye_pitch * effective_eye_weight + head_pitch * effective_head_weight

        gaze_confidence = 1.0
        h_span = self._last_h_span
        if abs(h_span) < 5.0:
            gaze_confidence = abs(h_span) / 5.0

        return raw_yaw, raw_pitch, head_yaw, head_pitch, gaze_confidence

    def _apply_calibration(self, raw_yaw: float, raw_pitch: float) -> tuple[float, float]:
        """Subtract calibration offsets from raw gaze angles.

        Args:
            raw_yaw: Uncalibrated yaw in degrees.
            raw_pitch: Uncalibrated pitch in degrees.

        Returns:
            ``(calibrated_yaw, calibrated_pitch)``.
        """
        calibrated_yaw = raw_yaw - self.yaw_offset
        calibrated_pitch = raw_pitch - self.pitch_offset
        return calibrated_yaw, calibrated_pitch

    def _stabilize(self, calibrated_yaw: float, calibrated_pitch: float,
                   dt: float) -> tuple[float, float]:
        """Apply adaptive EMA smoothing to reduce jitter.

        Uses a higher alpha during saccades (fast changes) and a lower alpha
        during fixations for smoother output.

        Args:
            calibrated_yaw: Calibrated yaw in degrees.
            calibrated_pitch: Calibrated pitch in degrees.
            dt: Frame delta time in seconds.

        Returns:
            ``(smooth_yaw, smooth_pitch)`` after EMA filtering.
        """
        if not self.initialized:
            self.smooth_yaw = calibrated_yaw
            self.smooth_pitch = calibrated_pitch
            self.initialized = True
        else:
            velocity = math.sqrt(
                ((calibrated_yaw - self.prev_raw_yaw) / dt) ** 2 +
                ((calibrated_pitch - self.prev_raw_pitch) / dt) ** 2
            ) if dt > 0 else 0

            alpha = (self.alpha_saccade if velocity > self.saccade_velocity
                     else self.alpha_fixation)

            self.smooth_yaw = alpha * calibrated_yaw + (1 - alpha) * self.smooth_yaw
            self.smooth_pitch = alpha * calibrated_pitch + (1 - alpha) * self.smooth_pitch

        self.prev_raw_yaw = calibrated_yaw
        self.prev_raw_pitch = calibrated_pitch

        return self.smooth_yaw, self.smooth_pitch

    def _project_and_classify(self, final_yaw: float, final_pitch: float,
                              raw_yaw: float, raw_pitch: float,
                              eye_yaw: float, eye_pitch: float,
                              head_yaw: float, head_pitch: float,
                              avg_iris_x: float, avg_iris_y: float,
                              gaze_confidence: float) -> dict[str, Any]:
        """Project gaze onto display coordinates and classify into a zone.

        Args:
            final_yaw: Stabilized yaw in degrees.
            final_pitch: Stabilized pitch in degrees.
            raw_yaw: Pre-stabilization yaw.
            raw_pitch: Pre-stabilization pitch.
            eye_yaw: Eye-only yaw component.
            eye_pitch: Eye-only pitch component.
            head_yaw: Head pose yaw.
            head_pitch: Head pose pitch.
            avg_iris_x: Averaged horizontal iris ratio.
            avg_iris_y: Averaged vertical iris ratio.
            gaze_confidence: Confidence ``[0, 1]`` based on eye aperture.

        Returns:
            Dict with keys: ``raw_yaw``, ``raw_pitch``, ``final_yaw``,
            ``final_pitch``, ``eye_yaw``, ``eye_pitch``, ``head_yaw``,
            ``head_pitch``, ``iris_x_avg``, ``iris_y_avg``, ``proj_x``,
            ``proj_y``, ``zone_id``, ``zone_name``, ``zone_type``,
            ``is_on_road``, ``gaze_confidence``.
        """
        origin_x = self.display_w / 2.0
        origin_y = self.display_h / 2.0
        proj_x = origin_x + final_yaw * self.ppd_yaw
        proj_y = origin_y + final_pitch * self.ppd_pitch

        zone_id: int = 15
        zone_name: str = 'UNKNOWN'
        zone_type: str = 'off_road'
        if self._zone_loader:
            zone_id, zone_name, zone_type = self._zone_loader.classify((proj_x, proj_y))

        return {
            'raw_yaw': raw_yaw,
            'raw_pitch': raw_pitch,
            'final_yaw': final_yaw,
            'final_pitch': final_pitch,
            'eye_yaw': eye_yaw,
            'eye_pitch': eye_pitch,
            'head_yaw': head_yaw,
            'head_pitch': head_pitch,
            'iris_x_avg': avg_iris_x,
            'iris_y_avg': avg_iris_y,
            'proj_x': proj_x,
            'proj_y': proj_y,
            'zone_id': zone_id,
            'zone_name': zone_name,
            'zone_type': zone_type,
            'is_on_road': zone_type == 'on_road',
            'gaze_confidence': gaze_confidence,
        }

    def estimate(self, landmarks: list[tuple[float, float]],
                 left_iris: tuple[float, float, float],
                 right_iris: tuple[float, float, float],
                 face_bbox: list[float],
                 dt: float = 0.04) -> dict[str, Any]:
        """Run the full gaze estimation pipeline for one frame.

        Args:
            landmarks: 25 facial landmarks in frame coordinates.
            left_iris: ``(cx, cy, r)`` for the left eye, or ``None``.
            right_iris: ``(cx, cy, r)`` for the right eye, or ``None``.
            face_bbox: ``[x1, y1, x2, y2]`` face bounding box.
            dt: Frame delta time in seconds for stabilization.

        Returns:
            Gaze result dict (see ``_project_and_classify`` return keys).
        """
        eye_yaw, eye_pitch, avg_iris_x, avg_iris_y = self._compute_iris_angles(
            landmarks, left_iris, right_iris)

        raw_yaw, raw_pitch, head_yaw, head_pitch, gaze_confidence = self._blend_with_head_pose(
            eye_yaw, eye_pitch, face_bbox, landmarks)

        calibrated_yaw, calibrated_pitch = self._apply_calibration(raw_yaw, raw_pitch)

        final_yaw, final_pitch = self._stabilize(calibrated_yaw, calibrated_pitch, dt)

        return self._project_and_classify(
            final_yaw, final_pitch, raw_yaw, raw_pitch,
            eye_yaw, eye_pitch, head_yaw, head_pitch,
            avg_iris_x, avg_iris_y, gaze_confidence)

    def reset(self) -> None:
        """Reset stabilization state and calibration offsets."""
        self.smooth_yaw = 0.0
        self.smooth_pitch = 0.0
        self.initialized = False
