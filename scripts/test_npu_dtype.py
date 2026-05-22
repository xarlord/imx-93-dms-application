#!/usr/bin/env python3
import sys
sys.path.insert(0, '/opt/dms')
import logging
logging.basicConfig(level=logging.DEBUG)

from src.utils.npu import NPUModel

print("Loading NPU model with ethosu...", flush=True)
m = NPUModel('/opt/gopoint-apps/downloads/face_detection_ptq_vela.tflite', use_npu=True)
print(f'Backend: {m.backend}', flush=True)
print(f'Input shape: {m.input_shape}', flush=True)
print(f'Input dtype: {m._input_dtype}', flush=True)
print(f'Quant scale: {m._quant_scale}', flush=True)
print(f'Quant zp: {m._quant_zero_point}', flush=True)
print(f'Has quantization: {m.has_quantization}', flush=True)

import numpy as np
inp = np.zeros((1, 128, 128, 3), dtype=np.float32)
inp = (inp - 0) * 128 + 128
inp = np.clip(inp, 0, 255).astype(np.float32)
inp = (inp - 128.0) / 128.0
print(f'Test input: shape={inp.shape}, dtype={inp.dtype}, range=[{inp.min():.2f}, {inp.max():.2f}]', flush=True)

try:
    out = m.predict(inp)
    print(f'Output: {len(out)} tensors', flush=True)
    for i, o in enumerate(out):
        print(f'  [{i}] shape={o.shape}, dtype={o.dtype}, range=[{o.min():.3f}, {o.max():.3f}]', flush=True)
except Exception as e:
    print(f'FAILED: {type(e).__name__}: {e}', flush=True)

print("Testing with uint8 input...", flush=True)
inp_u8 = np.zeros((1, 128, 128, 3), dtype=np.uint8)
try:
    out = m.predict(inp_u8)
    print(f'uint8 output: {len(out)} tensors', flush=True)
except Exception as e:
    print(f'uint8 FAILED: {type(e).__name__}: {e}', flush=True)
