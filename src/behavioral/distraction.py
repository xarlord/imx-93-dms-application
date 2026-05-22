"""Distraction detection: ADDW continuous timer + Euro NCAP VATS."""
import time
from collections import deque
from typing import Any, ClassVar


class DistractionDetector:
    """Detects distraction per ADDW and Euro NCAP VATS rules.
    ADDW: continuous off-road gaze (3.5s @ >=50km/h, 6.0s @ >=20km/h).
    VATS: cumulative 10s off-road in 30s window.
    Saccade tolerance: brief on-road glances <50ms don't reset timer.
    ADDW startup: 60s calibration period suppresses warnings per EU 2023/2590.
    """

    ADDW_CALIBRATION_SEC: ClassVar[float] = 60.0

    def __init__(self, saccade_tolerance_ms: float = 50,
                 vats_window_sec: float = 30,
                 vats_threshold_sec: float = 10.0,
                 fps: float = 25,
                 addw_activation_speed: int = 20,
                 gaze_confidence_threshold: float = 0.2) -> None:
        """Initializes the DistractionDetector.

        Args:
            saccade_tolerance_ms: Minimum on-road glance duration in milliseconds
                required to reset the continuous off-road timer.
            vats_window_sec: Size of the VATS cumulative sliding window in seconds.
            vats_threshold_sec: Cumulative off-road seconds within the VATS window
                that triggers a VATS event.
            fps: Expected frame rate for deque sizing.
            addw_activation_speed: Minimum vehicle speed in km/h for ADDW warnings.
            gaze_confidence_threshold: Minimum gaze confidence to consider gaze
                direction reliable.
        """
        self.saccade_tolerance: float = saccade_tolerance_ms / 1000.0
        self.vats_window_sec: float = vats_window_sec
        self.vats_threshold_sec: float = vats_threshold_sec
        self.frame_dt: float = 1.0 / fps
        self.addw_activation_speed: int = addw_activation_speed
        self.gaze_confidence_threshold: float = gaze_confidence_threshold

        self.offroad_timer: float = 0.0
        self.on_road_since: float | None = None

        self._start_time: float = time.monotonic()

        self.vats_window: deque[tuple[float, bool, float]] = deque(
            maxlen=int(vats_window_sec * fps + 10))

    def update(self, is_on_road: bool, vehicle_speed: float,
               dt: float | None = None,
               timestamp: float | None = None,
               gaze_confidence: float = 1.0) -> dict[str, Any]:
        """Processes one frame of gaze data and returns distraction indicators.

        Args:
            is_on_road: Whether gaze is classified as on-road.
            vehicle_speed: Current vehicle speed in km/h.
            dt: Time delta since last frame in seconds; defaults to ``1/fps``.
            timestamp: Monotonic timestamp override; defaults to ``time.monotonic()``.
            gaze_confidence: Confidence score for the gaze estimate (0.0-1.0).

        Returns:
            Dict with keys ``addw_nominal``, ``addw_buffer``, ``addw_fn_deadline``,
            ``vats_triggered``, ``vats_cumulative``, ``continuous_offroad``,
            ``is_distracted``, and ``in_calibration``.
        """
        now = timestamp or time.monotonic()
        dt = dt or self.frame_dt

        confident_off_road = (not is_on_road) and (gaze_confidence >= self.gaze_confidence_threshold)

        self.vats_window.append((now, confident_off_road, dt))

        if is_on_road and gaze_confidence >= self.gaze_confidence_threshold:
            if self.on_road_since is None:
                self.on_road_since = now
            sustained_on_road = now - self.on_road_since
            if sustained_on_road >= self.saccade_tolerance:
                self.offroad_timer = 0.0
        else:
            self.offroad_timer += dt
            self.on_road_since = None

        elapsed_since_start = now - self._start_time
        in_calibration = elapsed_since_start < self.ADDW_CALIBRATION_SEC

        addw_active = vehicle_speed >= self.addw_activation_speed
        if not addw_active or in_calibration:
            nominal = float('inf')
            buffer = float('inf')
            fn_deadline = float('inf')
        elif vehicle_speed >= 50:
            nominal = 3.5
            buffer = 5.0
            fn_deadline = 4.0
        else:
            nominal = 6.0
            buffer = 7.5
            fn_deadline = 6.5

        cutoff = now - self.vats_window_sec
        recent_offroad = sum(d for t, is_off, d in self.vats_window
                            if t > cutoff and is_off)

        vats_triggered = (addw_active and not in_calibration and
                          recent_offroad >= self.vats_threshold_sec)

        return {
            'addw_nominal': self.offroad_timer >= nominal,
            'addw_buffer': self.offroad_timer >= buffer,
            'addw_fn_deadline': self.offroad_timer >= fn_deadline,
            'vats_triggered': vats_triggered,
            'vats_cumulative': recent_offroad,
            'continuous_offroad': self.offroad_timer,
            'is_distracted': (self.offroad_timer >= nominal or vats_triggered),
            'in_calibration': in_calibration,
        }

    def reset(self) -> None:
        """Resets all timers, the VATS window, and restarts the calibration clock."""
        self.offroad_timer = 0.0
        self.vats_window.clear()
        self.on_road_since = None
        self._start_time = time.monotonic()

    def set_start_time(self, t: float) -> None:
        """Overrides the calibration start time.

        Args:
            t: Monotonic timestamp to use as the calibration epoch.
        """
        self._start_time = t
