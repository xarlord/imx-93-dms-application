"""Yawn detection via Mouth Aspect Ratio thresholding."""
import time


class YawnDetector:
    """Detects yawns from MAR threshold.
    Spec: MAR > 0.5 for >= 0.5 seconds.
    """

    def __init__(self, mar_threshold: float = 0.5,
                 min_duration: float = 0.5) -> None:
        """Initializes the YawnDetector.

        Args:
            mar_threshold: Mouth Aspect Ratio above which a yawn is considered active.
            min_duration: Minimum seconds above threshold to confirm a yawn.
        """
        self.mar_threshold: float = mar_threshold
        self.min_duration: float = min_duration
        self.yawn_start: float | None = None
        self.is_yawning: bool = False
        self.yawn_count: int = 0

    def update(self, mar: float, timestamp: float | None = None) -> None:
        """Processes one frame of Mouth Aspect Ratio for yawn detection.

        Args:
            mar: Current Mouth Aspect Ratio value.
            timestamp: Monotonic timestamp override; defaults to ``time.monotonic()``.
        """
        now = timestamp or time.monotonic()

        if mar > self.mar_threshold:
            if self.yawn_start is None:
                self.yawn_start = now
            duration = now - self.yawn_start
            if duration >= self.min_duration:
                self.is_yawning = True
        else:
            if self.is_yawning:
                self.yawn_count += 1
            self.is_yawning = False
            self.yawn_start = None

    def reset(self) -> None:
        """Resets all yawn tracking state."""
        self.yawn_start = None
        self.is_yawning = False
        self.yawn_count = 0
