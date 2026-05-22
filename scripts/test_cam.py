#!/usr/bin/env python3
import sys
sys.path.insert(0, '/opt/dms')
from src.capture import CameraCapture
import time

cap = CameraCapture(config={'device': '/dev/video0', 'width': 1280, 'height': 800, 'target_fps': 25})
count = [0]
last = [None]

def on_frame(f):
    count[0] += 1
    last[0] = f

cap.start(on_frame=on_frame)
time.sleep(5)
cap.stop()

print(f'Got {count[0]} frames in 5s = {count[0]/5:.1f} FPS')
if last[0] is not None:
    print(f'Shape: {last[0].shape}, dtype: {last[0].dtype}')
    print(f'Range: {last[0].min()} - {last[0].max()}')
else:
    print('No frames received')
