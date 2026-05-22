"""Drowsiness scoring: maps behavioral metrics to KSS level."""
import time
from typing import Any, ClassVar


class DrowsinessScorer:
    """Maps PERCLOS, blink rate, yawn, microsleep to KSS score and health metric.
    Spec: KSS >= 7 triggers advisory, KSS >= 8 triggers DDAW warning.
    Learning period: 10 minutes from journey start (Euro NCAP).
    """

    _RECOVERY_RATE: ClassVar[float] = 0.5
    _PENALTIES: ClassVar[dict[str, float]] = {
        'blink_elevated': 2.0,
        'yawn': 5.0,
        'microsleep': 15.0,
        'sleep': 30.0,
        'perclos_mild': 8.0,
        'perclos_severe': 20.0,
        'head_drop': 10.0,
    }

    def __init__(self, learning_period_sec: float = 600) -> None:
        """Initializes the DrowsinessScorer.

        Args:
            learning_period_sec: Duration in seconds of the initial learning period
                during which baseline adaptation occurs (Euro NCAP default: 600 s).
        """
        self.learning_start: float = time.monotonic()
        self.learning_period_sec: float = learning_period_sec
        self.health_score: float = 100.0

    def update(self, perclos_severity: str, blink_rate: float,
               yawn_active: bool, microsleep_state: str,
               head_yaw: float = 0, head_pitch: float = 0,
               timestamp: float | None = None) -> dict[str, Any]:
        """Updates the health score from behavioral signals and maps it to KSS.

        Args:
            perclos_severity: PERCLOS severity string (``'none'``, ``'mild'``,
                ``'severe'``).
            blink_rate: Estimated blinks per minute.
            yawn_active: Whether a yawn is currently in progress.
            microsleep_state: Current microsleep/sleep state string.
            head_yaw: Head yaw angle in degrees (currently unused).
            head_pitch: Head pitch angle in degrees.
            timestamp: Monotonic timestamp override; defaults to ``time.monotonic()``.

        Returns:
            Dict with keys ``health_score`` (0-100), ``kss`` (1-8), and
            ``in_learning_period`` (bool).
        """
        now = timestamp or time.monotonic()
        in_learning = (now - self.learning_start) < self.learning_period_sec

        if perclos_severity == 'severe':
            self.health_score -= self._PENALTIES['perclos_severe']
        elif perclos_severity == 'mild':
            self.health_score -= self._PENALTIES['perclos_mild']

        if microsleep_state in ('sleep', 'unresponsive'):
            self.health_score -= self._PENALTIES['sleep']
        elif microsleep_state == 'microsleep':
            self.health_score -= self._PENALTIES['microsleep']

        if yawn_active:
            self.health_score -= self._PENALTIES['yawn']

        if blink_rate > 25:
            self.health_score -= self._PENALTIES['blink_elevated']

        if abs(head_pitch) > 25:
            self.health_score -= self._PENALTIES['head_drop']

        self.health_score += self._RECOVERY_RATE
        self.health_score = max(0, min(100, self.health_score))

        kss = self._health_to_kss(self.health_score)

        return {
            'health_score': self.health_score,
            'kss': kss,
            'in_learning_period': in_learning,
        }

    @staticmethod
    def _health_to_kss(health: float) -> int:
        """Maps a health score to a Karolinska Sleepiness Scale level.

        Args:
            health: Health score in the range 0-100.

        Returns:
            Integer KSS level from 1 (alert) to 8 (very drowsy).
        """
        if health >= 85:
            return 1
        elif health >= 70:
            return 3
        elif health >= 55:
            return 5
        elif health >= 40:
            return 6
        elif health >= 25:
            return 7
        else:
            return 8

    def reset(self) -> None:
        """Resets the health score to 100 and restarts the learning period."""
        self.health_score = 100.0
        self.learning_start = time.monotonic()
