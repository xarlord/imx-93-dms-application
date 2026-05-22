"""PERCLOS drowsiness detection - percentage of eye closure over sliding window."""
import time
from collections import deque
from typing import ClassVar


class PERCLOSDetector:
    """PERCLOS p80: proportion of time eyes are >80% closed over a sliding window.
    Spec: 60-second window, >20% = severe (KSS >= 8), >10% = mild (KSS >= 7).
    Uses time-weighted integration rather than frame counting.
    """

    def __init__(self, window_sec: float = 60, close_threshold: float = 0.10) -> None:
        """Initializes the PERCLOSDetector.

        Args:
            window_sec: Size of the sliding window in seconds.
            close_threshold: Average EAR below which eyes are considered closed.
        """
        self.window_sec: float = window_sec
        self.close_threshold: float = close_threshold
        self.events: deque[tuple[float, bool]] = deque()

    def update(self, left_ear: float, right_ear: float,
               timestamp: float | None = None) -> None:
        """Records an eye-closure sample and prunes events outside the window.

        Args:
            left_ear: Eye Aspect Ratio for the left eye.
            right_ear: Eye Aspect Ratio for the right eye.
            timestamp: Monotonic timestamp override; defaults to ``time.monotonic()``.
        """
        now = timestamp or time.monotonic()
        avg_ear = (left_ear + right_ear) / 2.0
        is_closed = avg_ear < self.close_threshold
        self.events.append((now, is_closed))
        cutoff = now - self.window_sec
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def perclos(self) -> float:
        """Computes the time-weighted PERCLOS ratio over the sliding window.

        Returns:
            Fraction of the window during which eyes were closed (0.0-1.0).
        """
        if len(self.events) < 2:
            return 0.0
        total_time = 0.0
        closed_time = 0.0
        for i in range(1, len(self.events)):
            dt = self.events[i][0] - self.events[i - 1][0]
            total_time += dt
            if self.events[i - 1][1]:
                closed_time += dt
        return closed_time / total_time if total_time > 0 else 0.0

    def severity(self) -> str:
        """Classifies PERCLOS severity.

        Returns:
            ``'severe'`` if PERCLOS > 20%, ``'mild'`` if > 10%, else ``'none'``.
        """
        p = self.perclos()
        if p > 0.20:
            return 'severe'
        elif p > 0.10:
            return 'mild'
        return 'none'

    def reset(self) -> None:
        """Clears all recorded PERCLOS events."""
        self.events.clear()
