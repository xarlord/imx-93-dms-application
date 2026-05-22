#!/usr/bin/env python3
import signal
import sys
import time

import numpy as np


def handler(s, f):
    print("TIMEOUT after 15s", flush=True)
    sys.exit(1)


signal.signal(signal.SIGALRM, handler)
signal.alarm(15)

import tflite_runtime.interpreter as tflite

print("Loading delegate...", flush=True)
delegate = tflite.load_delegate("/usr/lib/libethosu_delegate.so")
print("Delegate loaded. Creating interpreter with Vela model...", flush=True)
interp = tflite.Interpreter(
    model_path="/opt/gopoint-apps/downloads/face_detection_ptq_vela.tflite",
    experimental_delegates=[delegate],
)
interp.allocate_tensors()
print("Interpreter OK!", flush=True)
details = interp.get_input_details()[0]
shape = details["shape"]
dtype = details["dtype"]
print(f"Input: shape={shape}, dtype={dtype}", flush=True)
inp = np.zeros((1, 128, 128, 3), dtype=np.uint8)
interp.set_tensor(details["index"], inp)
t0 = time.monotonic()
interp.invoke()
t1 = time.monotonic()
print(f"Inference: {(t1 - t0) * 1000:.1f} ms", flush=True)
out = interp.get_output_details()
for o in out:
    print(f"Output: shape={o['shape']}, dtype={o['dtype']}", flush=True)
signal.alarm(0)
print("DONE", flush=True)
