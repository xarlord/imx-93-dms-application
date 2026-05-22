#!/usr/bin/env python3
"""Run DMS application with real camera, real ML models, and HDMI display."""
import os
import signal
import sys
import time

sys.path.insert(0, '/opt/dms')
os.environ['XDG_RUNTIME_DIR'] = '/run/user/0'
os.environ['WAYLAND_DISPLAY'] = 'wayland-0'

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('dms')

from src.utils.config import DMSConfig
from src.pipeline import DMSPipeline
from src.capture import CameraCapture
from src.display import DisplayOutput


def main():
    cfg = DMSConfig('/opt/dms/config/config.yaml')
    cfg.set('models.use_npu', False)

    print('=' * 60)
    print('  DMS Driver Monitoring System - FRDM-IMX93')
    print('  Real Camera + Real ML Models (CPU) + HDMI Display')
    print('=' * 60)
    print()

    logger.info('Loading ML models...')
    pipe = DMSPipeline(cfg)
    logger.info('Pipeline ready')

    cam_cfg = cfg.camera
    capture = CameraCapture(config=cam_cfg)

    disp_cfg = cfg.display
    display = DisplayOutput(config=disp_cfg)

    logger.info('Starting HDMI display output...')
    display.start()
    logger.info('Display started')

    frame_count = [0]
    running = [True]

    def on_frame(frame_rgb):
        if not running[0]:
            return
        try:
            results = pipe.process_frame(frame_rgb)
            dashboard = pipe.render_dashboard(frame_rgb, results, fps=pipe.current_fps)
            display.push_frame(dashboard)

            frame_count[0] += 1
            if frame_count[0] % 25 == 0:
                kss = results.get('kss', 0)
                warn = results.get('warning_level', 'none')
                face = results.get('face_detected', False)
                zone = results.get('zone_name', '?')
                ear = results.get('ear', 0)
                fps = pipe.current_fps
                logger.info('Frame %d: FPS=%.1f KSS=%d warn=%s face=%s zone=%s EAR=%.2f',
                            frame_count[0], fps, kss, warn, face, zone, ear)
        except (RuntimeError, ValueError) as e:
            logger.error('Frame error: %s', e)

    logger.info('Starting camera capture...')
    capture.start(on_frame=on_frame)
    logger.info('Camera started - DMS running!')

    def signal_handler(sig, frame_arg):
        logger.info('Stopping DMS...')
        running[0] = False
        capture.stop()
        display.stop()
        logger.info('DMS stopped. %d frames processed.', frame_count[0])
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info('Press Ctrl+C to stop')
    try:
        while running[0]:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

    running[0] = False
    capture.stop()
    display.stop()
    logger.info('DMS stopped. %d frames processed.', frame_count[0])


if __name__ == '__main__':
    main()
