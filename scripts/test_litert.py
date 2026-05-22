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

print("Testing litert ethosu delegate...", flush=True)
delegate = tflite.load_delegate("/usr/lib/liblitert_ethosu_delegate.so")
print("Delegate loaded. Creating interpreter...", flush=True)
interp = tflite.Interpreter(
    model_path="/opt/gopoint-apps/downloads/face_detection_ptq_vela.tflite",
    experimental_delegates=[delegate],
)
interp.allocate_tensors()
print("Interpreter OK!", flush=True)
details = interp.get_input_details()[0]
print(f"Input: shape={details['shape']}, dtype={details['dtype']}", flush=True)
inp = np.zeros(tuple(int(x) for x in details["shape"]), dtype=details["dtype"])
interp.set_tensor(details["index"], inp)
t0 = time.monotonic()
for i in range(10):
    interp.invoke()
t1 = time.monotonic()
print(f"10 inferences: {(t1 - t0) * 1000:.1f} ms total, {(t1 - t0) / 10 * 1000:.1f} ms/inf", flush=True)
out_details = interp.get_output_details()
for o in out_details:
    arr = interp.get_tensor(o["index"])
    print(f"Output: shape={arr.shape}, dtype={arr.dtype}, range=[{arr.min():.3f}, {arr.max():.3f}]", flush=True)
signal.alarm(0)
print("DONE", flush=True)
