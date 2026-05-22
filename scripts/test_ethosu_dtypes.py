#!/usr/bin/env python3
import numpy as np

print("Testing ethosu.Interpreter input types...", flush=True)
import ethosu.interpreter as ethosu

interp = ethosu.Interpreter('/opt/gopoint-apps/downloads/face_detection_ptq_vela.tflite')
print("ethosu interpreter created", flush=True)

for dtype_name, dtype in [
    ('float32', np.float32),
    ('float64', np.float64),
    ('int8', np.int8),
    ('uint8', np.uint8),
    ('int16', np.int16),
    ('int32', np.int32),
]:
    try:
        inp = np.zeros((1, 128, 128, 3), dtype=dtype)
        interp.set_input(0, inp)
        interp.invoke()
        out = interp.get_output(0)
        print(f'  {dtype_name}: OK, output shape={np.array(out).shape}', flush=True)
    except Exception as e:
        print(f'  {dtype_name}: FAILED - {type(e).__name__}: {e}', flush=True)
