"""Dashboard renderer: composes 1280x800 BGRx frame with HUD overlays."""
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import cv2
    HAS_CV2: bool = True
except ImportError:
    HAS_CV2 = False


class DashboardRenderer:
    """Composes the DMS dashboard frame for HDMI display.

    Layout (1280x800): camera fills entire frame, status HUD overlaid on top.
    Works directly on BGRx (4-channel) frames — no format conversion needed.
    """

    def __init__(self, width: int = 1280, height: int = 800) -> None:
        self.width: int = width
        self.height: int = height
        self.frame: NDArray[np.uint8] = np.zeros((height, width, 4), dtype=np.uint8)
        self.frame[:, :, 3] = 255

    def render(self, camera_frame: NDArray[np.uint8] | None, results: dict[str, Any], fps: float = 0) -> NDArray[np.uint8]:
        """Compose full dashboard frame.

        Args:
            camera_frame: BGRx [H, W, 4] from camera (passed through directly).
            results: dict with all pipeline outputs
            fps: current FPS for display

        Returns:
            BGRx [H, W, 4] numpy array for display
        """
        if camera_frame is not None:
            fh, fw = camera_frame.shape[:2]
            if fh == self.height and fw == self.width and camera_frame.shape[2] == 4:
                self.frame[:] = camera_frame
            elif camera_frame.shape[2] == 3:
                import cv2
                resized = cv2.resize(camera_frame, (self.width, self.height))
                self.frame[:, :, :3] = resized
                self.frame[:, :, 3] = 255
            else:
                import cv2
                resized = cv2.resize(camera_frame, (self.width, self.height))
                self.frame[:] = resized
                self.frame[:, :, 3] = 255
        else:
            self.frame[:] = 0
            self.frame[:, :, 3] = 255

        self._draw_hud(results, fps)
        return self.frame

    def _draw_hud(self, results: dict[str, Any], fps: float) -> None:
        """Draw HUD overlay on top of camera frame."""
        if not HAS_CV2:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255, 255)
        green = (0, 255, 0, 255)
        red = (0, 0, 255, 255)
        yellow = (0, 255, 255, 255)
        gray = (180, 180, 180, 255)
        black = (0, 0, 0, 255)

        x, y = 10, 25
        lh = 22
        face = results.get('face_detected', False)
        kss = results.get('kss', 1)

        lines = [
            (f"FPS: {fps:.0f}", gray),
            (f"Face: {'YES' if face else 'NO'}", green if face else red),
            (f"KSS: {kss}", green if kss <= 3 else (yellow if kss <= 6 else red)),
        ]

        for text, color in lines:
            cv2.putText(self.frame, text, (x + 1, y + 1), font, 0.45, black, 1)
            cv2.putText(self.frame, text, (x, y), font, 0.45, color, 1)
            y += lh

        rx = self.width - 160
        ry = 25
        ear = results.get('ear', 0)
        perclos = results.get('perclos', 0)
        blink_rate = results.get('blink_rate', 0)
        zone = results.get('zone_name', 'UNKNOWN')

        metrics = [
            (f"EAR: {ear:.2f}", gray),
            (f"PERCLOS: {perclos:.1f}%", green if perclos < 10 else (yellow if perclos < 20 else red)),
            (f"Blink: {blink_rate:.0f}/m", gray),
            (f"Zone: {zone}", green if results.get('is_on_road', False) else yellow),
        ]

        for text, color in metrics:
            cv2.putText(self.frame, text, (rx + 1, ry + 1), font, 0.4, black, 1)
            cv2.putText(self.frame, text, (rx, ry), font, 0.4, color, 1)
            ry += lh

        warning = results.get('warning_level', 'none')
        if warning != 'none':
            bar_h = 30
            bar_y = self.height - bar_h
            w_colors = {
                'advisory': yellow,
                'escalating': (0, 165, 255, 255),
                'intervention': red,
                'emergency': red,
            }
            bar_color = w_colors.get(warning, green)
            cv2.rectangle(self.frame, (0, bar_y), (self.width, self.height), bar_color, -1)
            cv2.putText(self.frame, f"WARNING: {warning.upper()}", (20, bar_y + 22),
                        font, 0.7, white if warning != 'emergency' else black, 2)

    def get_bgrx_frame(self) -> NDArray[np.uint8]:
        """Return the current frame in BGRx format for GStreamer."""
        return self.frame
