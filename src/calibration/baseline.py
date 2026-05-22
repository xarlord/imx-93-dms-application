"""Driver baseline learning: establish normal EAR, blink rate during learning period."""
import time
from collections import deque
from typing import Any


class BaselineLearner:
    """Learns driver's baseline metrics during the learning period.

    Euro NCAP spec: up to 10 minutes from journey start.
    Collects: average open-eye EAR, average blink rate, average blink duration.
    Used to personalize thresholds after learning.
    """

    def __init__(self, learning_period_sec: int = 600, fps: int = 25) -> None:
        """Initializes the baseline learner with timing and default thresholds.

        Args:
            learning_period_sec: Duration in seconds to collect baseline samples.
            fps: Expected camera frame rate for rate calculations.
        """
        self.learning_period_sec: int = learning_period_sec
        self.fps: int = fps
        self.start_time: float | None = None
        self.ear_samples: deque[float] = deque(maxlen=fps * 30)
        self.blink_durations: deque[float] = deque(maxlen=200)
        self.total_frames: int = 0
        self.total_blinks: int = 0
        self.baseline_established: bool = False

        self.baseline_ear_open: float = 0.30
        self.baseline_blink_rate: float = 15.0
        self.baseline_blink_duration: float = 0.15

    def start(self, timestamp: float | None = None) -> None:
        """Marks the beginning of the learning period.

        Args:
            timestamp: Optional monotonic timestamp; defaults to now.
        """
        self.start_time = timestamp or time.monotonic()

    def update(self, ear_left: float, ear_right: float, is_blink: bool = False, timestamp: float | None = None) -> dict[str, Any]:
        """Feeds a new frame's metrics and returns learning-period status.

        Args:
            ear_left: Eye Aspect Ratio for the left eye.
            ear_right: Eye Aspect Ratio for the right eye.
            is_blink: Whether a blink was detected in this frame.
            timestamp: Optional monotonic timestamp; defaults to now.

        Returns:
            A dict with keys 'in_learning_period', 'elapsed_sec', and
            'baseline_established'.
        """
        now: float = timestamp or time.monotonic()
        if self.start_time is None:
            self.start_time = now

        self.total_frames += 1
        avg_ear: float = (ear_left + ear_right) / 2.0
        self.ear_samples.append(avg_ear)

        if is_blink:
            self.total_blinks += 1

        elapsed: float = now - self.start_time
        in_learning: bool = elapsed < self.learning_period_sec

        if not in_learning and not self.baseline_established:
            self._compute_baseline()

        return {
            'in_learning_period': in_learning,
            'elapsed_sec': elapsed,
            'baseline_established': self.baseline_established,
        }

    def _compute_baseline(self) -> None:
        if len(self.ear_samples) < 100:
            return

        sorted_ear: list[float] = sorted(self.ear_samples)
        q75_idx: int = int(len(sorted_ear) * 0.75)
        self.baseline_ear_open = sorted_ear[q75_idx]

        elapsed_min: float = self.total_frames / (self.fps * 60.0)
        if elapsed_min > 0:
            self.baseline_blink_rate = self.total_blinks / elapsed_min

        self.baseline_established = True

    @property
    def personalized_ear_close_threshold(self) -> float:
        """Personalized EAR close threshold based on driver's baseline."""
        return self.baseline_ear_open * 0.33

    @property
    def personalized_blink_rate_high(self) -> float:
        """Personalized high blink rate (elevated) threshold."""
        return self.baseline_blink_rate * 1.8

    def reset(self) -> None:
        """Clears all collected samples and resets the learner to initial state."""
        self.start_time = None
        self.ear_samples.clear()
        self.blink_durations.clear()
        self.total_frames = 0
        self.total_blinks = 0
        self.baseline_established = False
