#!/usr/bin/env python3
"""DMS Application: main entry point for the Driver Monitoring System.

Usage:
    python3 dms_app.py [--config CONFIG] [--mock] [--record FILE]

Arguments:
    --config CONFIG   Path to config.yaml (default: config/config.yaml)
    --mock            Run in mock mode (no models, synthetic data)
    --record FILE     Record metrics to CSV file
"""
import argparse
import logging
import os
import signal
import sys
import time
import threading
from typing import Any, TextIO

PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import DMSConfig
from src.pipeline import DMSPipeline
from src.capture import CameraCapture
from src.display import DisplayOutput

logger = logging.getLogger('dms.app')

LATENCY_WARN_MS: float = 500.0
LOG_EVERY: int = 100


def setup_logging(level: str = 'INFO', log_file: str | None = None) -> None:
    log_level: int = getattr(logging, level.upper(), logging.INFO)
    fmt: str = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    if log_file:
        logging.basicConfig(filename=log_file, level=log_level, format=fmt)
    else:
        logging.basicConfig(level=log_level, format=fmt)


class DMSApp:
    """Main DMS application coordinating capture, pipeline, and display."""

    def __init__(self, config_path: str, mock: bool = False) -> None:
        self.config: DMSConfig = DMSConfig(config_path)
        self.mock: bool = mock
        self._running: bool = False

        if mock:
            logger.info('Running in MOCK mode (no ML models)')

        self.pipeline: DMSPipeline = DMSPipeline(self.config)
        self.capture: CameraCapture = CameraCapture(config=self.config.camera)
        self.display: DisplayOutput = DisplayOutput(config=self.config.display)

        self._frame_count: int = 0
        self._start_time: float | None = None
        self._csv_file: TextIO | None = None
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._start_time = time.monotonic()
        logger.info('Starting DMS application')

        try:
            self.display.start()
            logger.info('Display started')
        except RuntimeError as e:
            logger.warning('Display not available: %s', e)

        try:
            self.capture.start()
            logger.info('Camera capture started')
        except RuntimeError as e:
            logger.error('Camera not available: %s', e)
            if self.mock:
                self._run_mock_loop()
            else:
                raise

        self._worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        self._running = False
        self.capture.stop()
        self.display.stop()
        elapsed: float = time.monotonic() - self._start_time if self._start_time else 0
        fps: float = self._frame_count / elapsed if elapsed > 0 else 0
        logger.info('DMS stopped. %d frames in %.1fs (%.1f fps)',
                      self._frame_count, elapsed, fps)
        if self._csv_file:
            self._csv_file.close()

    def _processing_loop(self) -> None:
        """Worker thread: pull latest frame, process, push to display.

        With block=true on appsrc, push_frame blocks until the display
        pipeline consumes the previous buffer. This naturally throttles
        the processing rate to match display throughput, keeping e2e
        latency at ~1 frame.
        """
        while self._running:
            frame = self.capture.get_frame(timeout=0.2)
            if frame is None:
                continue

            t_capture = time.monotonic()

            try:
                results: dict[str, Any] = self.pipeline.process_frame(frame)
                dashboard: Any = self.pipeline.render_dashboard(frame, results)
                self.display.push_frame(dashboard)

                t_end = time.monotonic()
                self._frame_count += 1

                e2e_ms = (t_end - t_capture) * 1000.0
                proc_fps = self.pipeline.current_fps

                if self._frame_count % LOG_EVERY == 0:
                    lat_status = "OK" if e2e_ms < LATENCY_WARN_MS else "SLOW"
                    face = results.get('face_detected', False)
                    bbox = results.get('head_bbox')
                    lm = results.get('landmarks')
                    logger.info(
                        'Frame %d | e2e=%.0fms [%s] | pipeline=%.0ffps | KSS=%d | %s | face=%s bbox=%s lm=%d',
                        self._frame_count, e2e_ms, lat_status,
                        proc_fps,
                        results.get('kss', 0),
                        results.get('warning_level', 'none'),
                        'Y' if face else 'N',
                        [int(v) for v in bbox] if bbox else '-',
                        len(lm) if lm else 0)

            except (RuntimeError, ValueError) as e:
                logger.error('Frame processing error: %s', e, exc_info=True)

    def _run_mock_loop(self) -> None:
        import numpy as np
        logger.info('Running mock processing loop')

        w: int = self.config.get('camera.width', 1280)
        h: int = self.config.get('camera.height', 800)
        dt: float = 1.0 / self.config.get('camera.target_fps', 25)

        try:
            while self._running:
                frame: Any = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
                results: dict[str, Any] = self.pipeline.process_frame(frame)
                dashboard: Any = self.pipeline.render_dashboard(frame, results)
                try:
                    self.display.push_frame(dashboard)
                except (RuntimeError, OSError):
                    pass
                self._frame_count += 1
                if self._frame_count % LOG_EVERY == 0:
                    logger.info('Frame %d, FPS=%.1f, KSS=%d, warning=%s',
                                 self._frame_count,
                                 self.pipeline.current_fps,
                                 results.get('kss', 0),
                                 results.get('warning_level', 'none'))
                time.sleep(dt)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='DMS Driver Monitoring System')
    parser.add_argument('--config', default=os.path.join(PROJECT_ROOT, 'config', 'config.yaml'),
                        help='Path to config.yaml')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode')
    parser.add_argument('--record', default=None, help='Record metrics to CSV file')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    args: argparse.Namespace = parser.parse_args()

    setup_logging(args.log_level)

    app: DMSApp = DMSApp(args.config, mock=args.mock)

    def signal_handler(sig: int, frame: Any) -> None:
        logger.info('Signal received, stopping...')
        app._running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app.start()

    try:
        while app._running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()


if __name__ == '__main__':
    main()
