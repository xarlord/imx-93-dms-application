"""9-point gaze calibration: maps known gaze directions to offset corrections."""
import time
import json
import os
from typing import Any


CALIBRATION_POINTS: list[tuple[str, float, float]] = [
    ('center', 0.0, 0.0),
    ('far_left', -30.0, 0.0),
    ('far_right', 30.0, 0.0),
    ('far_up', 0.0, -25.0),
    ('far_down', 0.0, 25.0),
    ('upper_left', -20.0, -15.0),
    ('upper_right', 20.0, -15.0),
    ('lower_left', -20.0, 15.0),
    ('lower_right', 20.0, 15.0),
]


class GazeCalibration:
    """9-point calibration for gaze offset correction.

    For each point, the driver gazes at a known location for a duration.
    The system records raw yaw/pitch and computes offsets so that
    (raw - offset) matches the expected direction.
    """

    def __init__(self, duration_per_point: float = 2.0, fps: int = 25) -> None:
        """Initializes the gaze calibration with timing parameters.

        Args:
            duration_per_point: Seconds to collect samples per calibration point.
            fps: Expected camera frame rate for sample count calculation.
        """
        self.duration_per_point: float = duration_per_point
        self.fps: int = fps
        self.current_point: int = 0
        self.collecting: bool = False
        self.samples: list[tuple[float, float]] = []
        self.results: list[dict[str, Any]] = []
        self.calibrated: bool = False

        self.yaw_offset: float = 0.0
        self.pitch_offset: float = 0.0

    def start(self) -> None:
        """Resets state and begins collecting samples for the first calibration point."""
        self.current_point = 0
        self.collecting = True
        self.samples = []
        self.results = []
        self.calibrated = False

    @property
    def current_label(self) -> str:
        """Returns the label of the current calibration point, or 'done'."""
        if self.current_point < len(CALIBRATION_POINTS):
            return CALIBRATION_POINTS[self.current_point][0]
        return 'done'

    @property
    def expected_yaw(self) -> float:
        """Returns the expected yaw angle for the current calibration point."""
        if self.current_point < len(CALIBRATION_POINTS):
            return CALIBRATION_POINTS[self.current_point][1]
        return 0.0

    @property
    def expected_pitch(self) -> float:
        """Returns the expected pitch angle for the current calibration point."""
        if self.current_point < len(CALIBRATION_POINTS):
            return CALIBRATION_POINTS[self.current_point][2]
        return 0.0

    @property
    def progress(self) -> str:
        """Returns a string showing calibration progress, e.g. '3/9'."""
        return f"{self.current_point + 1}/{len(CALIBRATION_POINTS)}"

    def update(self, raw_yaw: float, raw_pitch: float, timestamp: float | None = None) -> bool:
        """Feeds a raw gaze sample during calibration.

        Args:
            raw_yaw: Measured gaze yaw in degrees.
            raw_pitch: Measured gaze pitch in degrees.
            timestamp: Optional monotonic timestamp (unused, reserved).

        Returns:
            True when the current calibration point has finished collecting
            enough samples, False otherwise.
        """
        if not self.collecting or self.current_point >= len(CALIBRATION_POINTS):
            return True

        self.samples.append((raw_yaw, raw_pitch))

        frames_needed: int = int(self.duration_per_point * self.fps)
        if len(self.samples) >= frames_needed:
            avg_yaw: float = sum(s[0] for s in self.samples) / len(self.samples)
            avg_pitch: float = sum(s[1] for s in self.samples) / len(self.samples)

            expected: tuple[str, float, float] = CALIBRATION_POINTS[self.current_point]
            self.results.append({
                'label': expected[0],
                'expected_yaw': expected[1],
                'expected_pitch': expected[2],
                'measured_yaw': avg_yaw,
                'measured_pitch': avg_pitch,
            })

            self.current_point += 1
            self.samples = []

            if self.current_point >= len(CALIBRATION_POINTS):
                self._compute_offsets()
                self.collecting = False
                self.calibrated = True
            return True
        return False

    def _compute_offsets(self) -> None:
        """Compute yaw and pitch offsets from calibration data.

        For the center point (index 0), offset = measured - expected.
        Use all points for least-squares refinement if desired, but
        center point alone is sufficient for offset correction.
        """
        if not self.results:
            return

        center: dict[str, Any] = self.results[0]
        self.yaw_offset = center['measured_yaw'] - center['expected_yaw']
        self.pitch_offset = center['measured_pitch'] - center['expected_pitch']

    def save(self, filepath: str) -> None:
        """Writes calibration offsets and per-point results to a JSON file.

        Args:
            filepath: Destination file path for the JSON output.
        """
        data: dict[str, Any] = {
            'yaw_offset': self.yaw_offset,
            'pitch_offset': self.pitch_offset,
            'points': self.results,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> bool:
        """Loads previously saved calibration offsets from a JSON file.

        Args:
            filepath: Path to the JSON calibration file.

        Returns:
            True if the file was loaded successfully, False if it does not exist.
        """
        if not os.path.exists(filepath):
            return False
        with open(filepath, 'r', encoding='utf-8') as f:
            data: dict[str, Any] = json.load(f)
        self.yaw_offset = data.get('yaw_offset', 0.0)
        self.pitch_offset = data.get('pitch_offset', 0.0)
        self.calibrated = True
        return True
