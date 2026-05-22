#!/usr/bin/env python3
"""Quick pipeline test on board with real camera (mock models)."""
import sys, os, time
sys.path.insert(0, '/opt/dms')
import numpy as np

from src.utils.config import DMSConfig
from src.pipeline import DMSPipeline
from src.capture import CameraCapture

cfg = DMSConfig('/opt/dms/config/config.yaml')
cfg._flat['models.mock'] = True
cfg._flat['models.head_detector'] = ''
cfg._flat['models.landmark'] = ''
cfg._flat['models.iris'] = ''

pipe = DMSPipeline(cfg)
print('Pipeline loaded')

cap = CameraCapture(config=cfg.camera)
frame_count = [0]

def on_frame(frame):
    try:
        results = pipe.process_frame(frame)
        frame_count[0] += 1
        kss = results.get('kss', 0)
        warn = results.get('warning_level', 'none')
        face = results.get('face_detected', False)
        zone = results.get('zone_name', '?')
        fps = pipe.current_fps
        if frame_count[0] <= 5 or frame_count[0] % 10 == 0:
            print('Frame %d: FPS=%.1f KSS=%d warning=%s face=%s zone=%s' %
                  (frame_count[0], fps, kss, warn, face, zone))
        if frame_count[0] >= 25:
            cap.stop()
    except Exception as e:
        import traceback
        traceback.print_exc()
        cap.stop()

cap.start(on_frame=on_frame)

for i in range(300):
    if frame_count[0] >= 25 or not cap.running:
        break
    time.sleep(0.1)

if cap.running:
    cap.stop()

print()
print('=== %d frames processed, FPS=%.1f ===' % (frame_count[0], pipe.current_fps))
