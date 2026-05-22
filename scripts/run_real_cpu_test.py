#!/usr/bin/env python3
"""Test full DMS pipeline with REAL ML models (CPU) + real camera on i.MX93."""
import sys, os
sys.path.insert(0, '/opt/dms')
os.environ['DMS_FORCE_CPU'] = '1'
import time
import numpy as np

from src.utils.config import DMSConfig
from src.pipeline import DMSPipeline
from src.capture import CameraCapture

cfg = DMSConfig('/opt/dms/config/config.yaml')
# Force CPU inference (models not Vela-optimized yet)
cfg._flat['models.use_npu'] = False

print('=== DMS Pipeline: Real ML Models (CPU) + Real Camera ===')
print()

pipe = DMSPipeline(cfg)
print('Pipeline loaded (CPU inference)')

cap = CameraCapture(config=cfg.camera)
frame_count = [0]
errors = [0]

def on_frame(frame):
    try:
        t0 = time.monotonic()
        results = pipe.process_frame(frame)
        dt = (time.monotonic() - t0) * 1000
        frame_count[0] += 1

        kss = results.get('kss', 0)
        warn = results.get('warning_level', 'none')
        face = results.get('face_detected', False)
        zone = results.get('zone_name', '?')
        ear = results.get('ear', 0)

        if frame_count[0] <= 10 or frame_count[0] % 10 == 0:
            print('Frame %3d: %5.1fms | KSS=%d warn=%-10s face=%s zone=%-12s EAR=%.2f' %
                  (frame_count[0], dt, kss, warn, face, zone, ear))

        if frame_count[0] >= 25:
            cap.stop()
    except Exception as e:
        import traceback
        errors[0] += 1
        if errors[0] <= 3:
            traceback.print_exc()
        if errors[0] >= 5:
            cap.stop()

cap.start(on_frame=on_frame)

for i in range(300):
    if frame_count[0] >= 25 or not cap.running:
        break
    time.sleep(0.1)

if cap.running:
    cap.stop()

print()
print('Frames: %d | Errors: %d | FPS: %.1f' % (frame_count[0], errors[0], pipe.current_fps))
if errors[0] == 0 and frame_count[0] >= 5:
    print('SUCCESS')
