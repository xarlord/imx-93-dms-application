#!/usr/bin/env python3
import time
import sys

import numpy as np

sys.path.insert(0, "/opt/dms")
from src.capture import CameraCapture
from src.utils.config import DMSConfig

cfg = DMSConfig("/opt/dms/config/config.yaml")
cam_cfg = cfg.camera

print(f"Camera config: {cam_cfg}", flush=True)

frame_count = 0
t0 = 0


def on_frame(frame_rgb):
    global frame_count, t0
    frame_count += 1
    if frame_count == 1:
        t0 = time.monotonic()
        print(f"First frame: shape={frame_rgb.shape}, dtype={frame_rgb.dtype}", flush=True)
    if frame_count % 50 == 0:
        elapsed = time.monotonic() - t0
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"Frame {frame_count}: {fps:.1f} FPS", flush=True)


cap = CameraCapture(config=cam_cfg)
print("Starting camera...", flush=True)
cap.start(on_frame=on_frame)
print("Capturing for 5s...", flush=True)
time.sleep(5)
cap.stop()

elapsed = time.monotonic() - t0
fps = frame_count / elapsed if elapsed > 0 else 0
print(f"Total: {frame_count} frames in {elapsed:.1f}s = {fps:.1f} FPS", flush=True)
print("CAMERA TEST DONE", flush=True)
