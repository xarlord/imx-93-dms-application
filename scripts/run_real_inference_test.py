#!/usr/bin/env python3
"""Test full DMS pipeline with REAL ML models + real camera on i.MX93."""
import sys
sys.path.insert(0, '/opt/dms')
import time
import numpy as np

from src.utils.config import DMSConfig
from src.pipeline import DMSPipeline
from src.capture import CameraCapture

cfg = DMSConfig('/opt/dms/config/config.yaml')
# Use REAL models (not mock)

print('=== DMS Pipeline Test: Real ML Models + Real Camera ===')
print()

pipe = DMSPipeline(cfg)
print('Pipeline loaded with real ML models')

cap = CameraCapture(config=cfg.camera)
frame_count = [0]
errors = [0]

def on_frame(frame):
    try:
        t0 = time.monotonic()
        results = pipe.process_frame(frame)
        dt = time.monotonic() - t0
        frame_count[0] += 1

        kss = results.get('kss', 0)
        warn = results.get('warning_level', 'none')
        face = results.get('face_detected', False)
        zone = results.get('zone_name', '?')
        fps = pipe.current_fps
        ear = results.get('ear', 0)
        perclos = results.get('perclos', 0)

        if frame_count[0] <= 10 or frame_count[0] % 10 == 0:
            print('Frame %3d: %.1fms | KSS=%d warn=%-10s face=%s zone=%-12s EAR=%.2f PERCLOS=%.1f%%' %
                  (frame_count[0], dt*1000, kss, warn, face, zone, ear, perclos*100))

        if frame_count[0] >= 30:
            cap.stop()
    except Exception as e:
        import traceback
        errors[0] += 1
        if errors[0] <= 3:
            print('ERROR frame %d:' % frame_count[0])
            traceback.print_exc()
        if errors[0] >= 5:
            cap.stop()

cap.start(on_frame=on_frame)

for i in range(300):
    if frame_count[0] >= 30 or not cap.running:
        break
    time.sleep(0.1)

if cap.running:
    cap.stop()

print()
print('=== Results ===')
print('Frames processed: %d' % frame_count[0])
print('Errors: %d' % errors[0])
print('Pipeline FPS: %.1f' % pipe.current_fps)
if errors[0] == 0 and frame_count[0] >= 10:
    print('SUCCESS: Pipeline running with real ML models')
else:
    print('ISSUES: See errors above')
