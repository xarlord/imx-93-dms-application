#!/usr/bin/env python3
import signal
import sys
import time

import numpy as np


def handler(s, f):
    print("TIMEOUT after 10s", flush=True)
    sys.exit(1)


signal.signal(signal.SIGALRM, handler)

print("Step 1: import ethosu...", flush=True)
import ethosu.interpreter as ethosu

print("Step 2: creating interpreter...", flush=True)
signal.alarm(10)
interp = ethosu.Interpreter("/opt/gopoint-apps/downloads/face_detection_ptq_vela.tflite")
signal.alarm(0)
print("Step 3: interpreter created!", flush=True)

print("Step 4: set_input...", flush=True)
inp = np.zeros((1, 128, 128, 3), dtype=np.uint8)
interp.set_input(0, inp)

print("Step 5: invoke...", flush=True)
signal.alarm(10)
t0 = time.monotonic()
interp.invoke()
t1 = time.monotonic()
signal.alarm(0)
print(f"Step 6: inference done in {(t1 - t0) * 1000:.1f} ms", flush=True)

for i in range(4):
    try:
        out = interp.get_output(i)
        print(f"Output {i}: shape={np.array(out).shape}", flush=True)
    except Exception as e:
        print(f"Output {i}: {e}", flush=True)
        break

print("ALL DONE", flush=True)
