"""Warning visual rendering: overlay warning indicators on dashboard."""
try:
    import cv2
    HAS_CV2: bool = True
except ImportError:
    HAS_CV2 = False

import time
import numpy as np
from numpy.typing import NDArray


class WarningVisuals:
    """Renders warning-level visual indicators (flashing, pulsing overlays)."""

    def __init__(self) -> None:
        """Initialize the WarningVisuals flash state."""
        self._flash_state: bool = False
        self._last_toggle: float = 0.0

    def render(self, frame: NDArray[np.uint8], warning_level: str, timestamp: float | None = None) -> None:
        """Draw warning overlay on frame with flash/pulse effects.

        Renders level-appropriate visual indicators:
        ``emergency`` (red tint + text), ``intervention`` (light red tint),
        ``escalating`` (yellow border), ``advisory`` (thin yellow border).

        Args:
            frame: BGR numpy array to draw on (modified in place).
            warning_level: One of ``'none'``, ``'advisory'``, ``'escalating'``,
                ``'intervention'``, ``'emergency'``.
            timestamp: Optional monotonic timestamp; defaults to
                ``time.monotonic()``.
        """
        if not HAS_CV2:
            return

        now: float = timestamp or time.monotonic()

        # Flash toggle at ~4Hz
        if now - self._last_toggle > 0.125:
            self._flash_state = not self._flash_state
            self._last_toggle = now

        h: int
        w: int
        h, w = frame.shape[:2]

        if warning_level == 'emergency':
            if self._flash_state:
                overlay: NDArray[np.uint8] = frame.copy()
                overlay[:, :, :3] = (0, 0, 200)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Large warning text
            cv2.putText(frame, "!! EMERGENCY !!", (w // 2 - 200, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        elif warning_level == 'intervention':
            if self._flash_state:
                overlay = frame.copy()
                overlay[:, :, :3] = (0, 0, 200)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            cv2.putText(frame, "WARNING: INTERVENTION", (w // 2 - 250, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        elif warning_level == 'escalating':
            # Pulsing yellow border
            if self._flash_state:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 255), 4)

            cv2.putText(frame, "ESCALATING WARNING", (w // 2 - 200, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        elif warning_level == 'advisory':
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 255), 2)
            cv2.putText(frame, "Advisory", (w // 2 - 60, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
