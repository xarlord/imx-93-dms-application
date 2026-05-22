"""Microsleep and sleep detection from eye closure duration."""
import time


class MicrosleepDetector:
    """Detects eye closures by duration.
    Spec: 1-2s = microsleep, >=3s = sleep, >=6s = unresponsive.
    """

    def __init__(self, ear_close_threshold: float = 0.10) -> None:
        """Initializes the MicrosleepDetector.

        Args:
            ear_close_threshold: Average EAR below which eyes are considered closed.
        """
        self.ear_close_threshold: float = ear_close_threshold
        self.eyes_closed_since: float | None = None
        self.state: str = 'open'
        self._last_microsleep_time: float | None = None
        self._last_sleep_time: float | None = None
        self.last_event: str | None = None
        self.last_event_time: float | None = None

    def update(self, left_ear: float, right_ear: float,
               timestamp: float | None = None) -> str:
        """Evaluates eye closure state from EAR values.

        Args:
            left_ear: Eye Aspect Ratio for the left eye.
            right_ear: Eye Aspect Ratio for the right eye.
            timestamp: Monotonic timestamp override; defaults to ``time.monotonic()``.

        Returns:
            Current state string: ``'open'``, ``'closed'``, ``'microsleep'``,
            ``'sleep'``, or ``'unresponsive'``.
        """
        now = timestamp or time.monotonic()
        avg_ear = (left_ear + right_ear) / 2.0
        is_closed = avg_ear < self.ear_close_threshold

        if is_closed:
            if self.eyes_closed_since is None:
                self.eyes_closed_since = now
            duration = now - self.eyes_closed_since

            if duration >= 6.0:
                self.state = 'unresponsive'
                self.last_event = 'unresponsive'
                self.last_event_time = now
            elif duration >= 3.0:
                self.state = 'sleep'
                self._last_sleep_time = now
                self.last_event = 'sleep'
                self.last_event_time = now
            elif duration >= 1.0:
                self.state = 'microsleep'
                self._last_microsleep_time = now
                self.last_event = 'microsleep'
                self.last_event_time = now
            else:
                self.state = 'closed'
        else:
            self.eyes_closed_since = None
            self.state = 'open'

        return self.state

    @property
    def closure_duration(self) -> float:
        """Returns the duration of the current eye closure in seconds.

        Returns:
            Seconds since eyes closed, or 0.0 if eyes are open.
        """
        if self.eyes_closed_since is None:
            return 0.0
        return time.monotonic() - self.eyes_closed_since

    def reset(self) -> None:
        """Resets the detector to the initial open-eyes state."""
        self.eyes_closed_since = None
        self.state = 'open'
        self.last_event = None
        self.last_event_time = None
