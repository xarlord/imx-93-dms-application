"""GStreamer display output: appsrc -> PXP CSC -> waylandsink for HDMI."""
import logging
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    HAS_GST: bool = True
except (ImportError, ValueError):
    HAS_GST = False

logger = logging.getLogger('dms.display')

PIPELINE_DISPLAY: str = (
    "appsrc name=src format=GST_FORMAT_TIME block=true max-buffers=1 ! "
    "video/x-raw,format=BGRx,width={width},height={height},framerate={fps}/1 ! "
    "imxvideoconvert_pxp ! "
    "waylandsink fullscreen={fullscreen} sync=false"
)


class DisplayOutput:
    """GStreamer display output pushing BGRx frames to HDMI via waylandsink.

    Uses block=true on appsrc to prevent frame backlog — the producer
    (processing thread) is throttled to the display consumption rate,
    ensuring zero lag between capture and display.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.width: int = (config or {}).get('width', 1280)
        self.height: int = (config or {}).get('height', 800)
        self.target_fps: int = (config or {}).get('target_fps', 25)
        self.fullscreen: bool = (config or {}).get('fullscreen', True)

        self._pipeline: Any = None
        self._appsrc: Any = None
        self._bus: Any = None
        self._loop: Any = None
        self._thread: threading.Thread | None = None
        self._running: bool = False
        self._frame_count: int = 0
        self._drop_count: int = 0
        self._pts: int = 0
        self._duration: int | None = None

    def start(self) -> None:
        """Start the display pipeline."""
        if not HAS_GST:
            raise RuntimeError('GStreamer not available')

        Gst.init(None)
        self._running = True

        pipeline_str: str = PIPELINE_DISPLAY.format(
            width=self.width,
            height=self.height,
            fps=self.target_fps,
            fullscreen='true' if self.fullscreen else 'false',
        )
        logger.info('Starting display pipeline: %s', pipeline_str)

        self._pipeline = Gst.parse_launch(pipeline_str)
        self._appsrc = self._pipeline.get_by_name('src')

        self._duration = int(1e9 / self.target_fps)

        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect('message::error', self._on_error)

        self._pipeline.set_state(Gst.State.PLAYING)

        self._loop = GLib.MainLoop()
        self._thread = threading.Thread(target=self._loop.run, daemon=True)
        self._thread.start()

        logger.info('Display output started')

    def stop(self) -> None:
        """Stop the display pipeline."""
        self._running = False
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        if self._loop and self._loop.is_running():
            self._loop.quit()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info('Display output stopped (pushed=%d, dropped=%d)',
                     self._frame_count, self._drop_count)

    def push_frame(self, frame: NDArray[np.uint8]) -> bool:
        """Push a BGRx frame to the display pipeline (blocks if pipeline busy).

        With block=true on appsrc, this call blocks until the downstream
        pipeline has consumed the previous buffer. This prevents frame
        accumulation and ensures display latency stays at 1 frame.

        Args:
            frame: numpy array [H, W, 4] uint8 BGRx.

        Returns:
            True if frame was pushed, False on error.
        """
        if not self._running or self._appsrc is None:
            return False

        expected_shape: tuple[int, int, int] = (self.height, self.width, 4)
        if frame.shape != expected_shape:
            logger.warning('Frame shape mismatch: expected %s, got %s',
                           expected_shape, frame.shape)
            return False

        buf: Any = Gst.Buffer.new_wrapped(frame.tobytes())
        buf.pts = self._pts
        buf.duration = self._duration
        self._pts += self._duration

        ret: Any = self._appsrc.emit('push-buffer', buf)
        if ret == Gst.FlowReturn.OK:
            self._frame_count += 1
            return True
        else:
            self._drop_count += 1
            return False

    @property
    def frame_count(self) -> int:
        """Returns the total number of frames pushed to the display."""
        return self._frame_count

    @property
    def running(self) -> bool:
        """Returns True if the display pipeline is currently active."""
        return self._running

    def _on_error(self, bus: Any, msg: Any) -> None:
        err: Any
        debug: str
        err, debug = msg.parse_error()
        logger.error('GStreamer display error: %s (%s)', err, debug)
        self._running = False
