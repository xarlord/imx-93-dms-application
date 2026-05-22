"""Blink detection via Eye Aspect Ratio thresholding."""
import time
from collections import deque


class BlinkDetector:
    """Detects blinks from EAR crossing threshold.
    Spec: EAR_BLINK_THRESHOLD = 0.19, tracks blink rate per minute.
    """

    def __init__(self, blink_threshold: float = 0.19,
                 min_duration_ms: float = 50) -> None:
        """Initializes the BlinkDetector.

        Args:
            blink_threshold: Average EAR below which a blink is considered to start.
            min_duration_ms: Minimum blink duration in milliseconds to count as valid.
        """
        self.blink_threshold: float = blink_threshold
        self.min_duration_ms: float = min_duration_ms
        self.in_blink: bool = False
        self.blink_start: float | None = None
        self.blink_durations: deque[tuple[float, float]] = deque(maxlen=200)
        self.blink_count: int = 0

    def update(self, left_ear: float, right_ear: float,
               timestamp: float | None = None) -> None:
        """Processes one frame of EAR values for blink detection.

        Args:
            left_ear: Eye Aspect Ratio for the left eye.
            right_ear: Eye Aspect Ratio for the right eye.
            timestamp: Monotonic timestamp override; defaults to ``time.monotonic()``.
        """
        now = timestamp or time.monotonic()
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < self.blink_threshold:
            if not self.in_blink:
                self.in_blink = True
                self.blink_start = now
        else:
            if self.in_blink:
                duration_ms = (now - self.blink_start) * 1000
                if duration_ms >= self.min_duration_ms:
                    self.blink_durations.append((now, duration_ms))
                    self.blink_count += 1
                self.in_blink = False
                self.blink_start = None

    def blink_rate(self, window_sec: float = 60,
                   timestamp: float | None = None) -> float:
        """Computes the blink rate normalized to blinks per minute.

        Args:
            window_sec: Look-back window in seconds.
            timestamp: Monotonic timestamp override; defaults to ``time.monotonic()``.

        Returns:
            Estimated blinks per minute within the window.
        """
        now = timestamp or time.monotonic()
        cutoff = now - window_sec
        recent = sum(1 for t, _ in self.blink_durations if t > cutoff)
        return recent * (60.0 / window_sec)

    @property
    def is_blinking(self) -> bool:
        """Returns whether a blink is currently in progress.

        Returns:
            True if the detector is mid-blink.
        """
        return self.in_blink

    def reset(self) -> None:
        """Resets all blink tracking state."""
        self.in_blink = False
        self.blink_start = None
        self.blink_durations.clear()
        self.blink_count = 0
