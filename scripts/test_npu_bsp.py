#!/usr/bin/env python3
import signal
import sys
import time

import numpy as np


def handler(s, f):
    print("TIMEOUT after 15s", flush=True)
    sys.exit(1)


signal.signal(signal.SIGALRM, handler)
signal.alarm(30)

print("Step 1: importing tflite...", flush=True)
import tflite_runtime.interpreter as tflite

print("Step 2: loading ethosu delegate...", flush=True)
delegate = tflite.load_delegate("/usr/lib/libethosu_delegate.so")
print("Step 3: delegate loaded OK", flush=True)

vela_model = "/opt/gopoint-apps/downloads/face_detection_ptq_vela.tflite"
print(f"Step 4: creating interpreter with {vela_model}...", flush=True)
interp = tflite.Interpreter(
    model_path=vela_model,
    experimental_delegates=[delegate],
)
print("Step 5: allocating tensors...", flush=True)
interp.allocate_tensors()
print("Step 6: interpreter ready!", flush=True)

details = interp.get_input_details()[0]
print(f"Input: shape={details['shape']}, dtype={details['dtype']}", flush=True)

inp = np.zeros(tuple(int(x) for x in details["shape"]), dtype=details["dtype"])
interp.set_tensor(details["index"], inp)

print("Step 7: running inference...", flush=True)
t0 = time.monotonic()
interp.invoke()
t1 = time.monotonic()
print(f"First inference: {(t1 - t0) * 1000:.1f} ms", flush=True)

t0 = time.monotonic()
for i in range(10):
    interp.invoke()
t1 = time.monotonic()
print(f"10 inferences: {(t1 - t0) * 1000:.1f} ms total, {(t1 - t0) / 10 * 1000:.1f} ms/inf", flush=True)

out_details = interp.get_output_details()
for o in out_details:
    arr = interp.get_tensor(o["index"])
    print(f"Output: shape={arr.shape}, dtype={arr.dtype}", flush=True)

signal.alarm(0)
print("ALL DONE", flush=True)
