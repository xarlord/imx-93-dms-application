"""GStreamer camera capture: v4l2src -> PXP CSC -> appsink -> numpy frames."""
import logging
import queue
import threading
import time
from typing import Any, Callable, Self

import numpy as np
from numpy.typing import NDArray

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    HAS_GST: bool = True
except (ImportError, ValueError):
    HAS_GST = False

logger = logging.getLogger('dms.capture')

PIPELINE_CAPTURE: str = (
    "v4l2src device={device} ! "
    "video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1 ! "
    "imxvideoconvert_pxp ! "
    "video/x-raw,format=BGRx ! "
    "appsink name=sink max-buffers=2 drop=true emit-signals=true"
)


class CameraCapture:
    """GStreamer camera capture producing contiguous BGRx numpy frames.

    Runs a GStreamer pipeline: v4l2src -> PXP CSC -> appsink.
    Frames are delivered via a queue for decoupled processing.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.device: str = (config or {}).get('device', '/dev/video0')
        self.width: int = (config or {}).get('width', 1280)
        self.height: int = (config or {}).get('height', 800)
        self.target_fps: int = (config or {}).get('target_fps', 25)

        self._pipeline: Any = None
        self._bus: Any = None
        self._loop: Any = None
        self._thread: threading.Thread | None = None
        self._frame_count: int = 0
        self._running: bool = False
        self._frame_queue: queue.Queue[NDArray[np.uint8]] = queue.Queue(maxsize=2)

        self._target_brightness: float = 100.0
        self._wb_alpha: float = 0.05
        self._current_gains: NDArray[np.float32] = np.array([1.4, 1.0, 0.8], dtype=np.float32)
        self._rebuild_lut()

    def _rebuild_lut(self) -> None:
        lut = np.arange(256, dtype=np.float32)
        self._wb_lut: NDArray[np.uint8] = np.stack([
            np.clip(lut * self._current_gains[i], 0, 255).astype(np.uint8) for i in range(3)
        ], axis=-1)

    def _adapt_gains(self, frame: NDArray[np.uint8]) -> None:
        bgr = frame[:, :, :3]
        means = bgr.mean(axis=(0, 1)).astype(np.float32)
        avg_brightness = means.mean()
        if avg_brightness < 1.0:
            return
        for c in range(3):
            target_c = self._target_brightness * (means[c] / avg_brightness)
            if means[c] > 1.0:
                ideal_gain = target_c / means[c]
                self._current_gains[c] += self._wb_alpha * (ideal_gain - self._current_gains[c])
                self._current_gains[c] = np.clip(self._current_gains[c], 0.5, 3.0)
        self._rebuild_lut()

    def start(self, on_frame: Callable[[NDArray[np.uint8]], None] | None = None) -> None:
        """Start the capture pipeline.

        Args:
            on_frame: Unused, kept for API compatibility.
        """
        if not HAS_GST:
            raise RuntimeError('GStreamer not available')

        Gst.init(None)
        self._running = True

        pipeline_str: str = PIPELINE_CAPTURE.format(
            device=self.device,
            width=self.width,
            height=self.height,
            fps=self.target_fps,
        )
        logger.info('Starting capture pipeline: %s', pipeline_str)

        self._pipeline = Gst.parse_launch(pipeline_str)
        appsink: Any = self._pipeline.get_by_name('sink')
        appsink.connect('new-sample', self._on_new_sample)

        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect('message::error', self._on_error)

        self._pipeline.set_state(Gst.State.PLAYING)

        self._loop = GLib.MainLoop()
        self._thread = threading.Thread(target=self._loop.run, daemon=True)
        self._thread.start()

        logger.info('Camera capture started')

    def stop(self) -> None:
        """Stop the capture pipeline."""
        self._running = False
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        if self._loop and self._loop.is_running():
            self._loop.quit()
        if self._thread:
            self._thread.join(timeout=3.0)
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        logger.info('Camera capture stopped')

    def get_frame(self, timeout: float = 0.1) -> NDArray[np.uint8] | None:
        """Get the latest captured frame (blocking with timeout).

        Args:
            timeout: Max seconds to wait for a frame.

        Returns:
            numpy array [H, W, 4] uint8 BGRx (contiguous), or None.
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def frame_count(self) -> int:
        """Returns the total number of frames captured since start."""
        return self._frame_count

    @property
    def running(self) -> bool:
        """Returns True if the capture pipeline is currently active."""
        return self._running

    def _on_new_sample(self, appsink: Any) -> Any:
        """GStreamer appsink callback: extract frame as contiguous BGRx array."""
        sample: Any = appsink.emit('pull-sample')
        if not sample:
            return Gst.FlowReturn.OK

        buf: Any = sample.get_buffer()
        success: bool
        info: Any
        success, info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        try:
            frame: NDArray[np.uint8] = np.frombuffer(info.data, dtype=np.uint8)
            frame = frame.reshape((self.height, self.width, 4)).copy()
            if self._frame_count % 5 == 0:
                self._adapt_gains(frame)
            bgr = frame[:, :, :3]
            for c in range(3):
                bgr[:, :, c] = self._wb_lut[:, c].take(bgr[:, :, c])
            self._frame_count += 1

            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            self._frame_queue.put_nowait(frame)
        except (ValueError, RuntimeError, queue.Full) as e:
            logger.error('Frame extraction error: %s', e)
        finally:
            buf.unmap(info)

        return Gst.FlowReturn.OK

    def _on_error(self, bus: Any, msg: Any) -> None:
        err: Any
        debug: str
        err, debug = msg.parse_error()
        logger.error('GStreamer error: %s (%s)', err, debug)
        self._running = False
