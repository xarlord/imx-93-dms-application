#!/usr/bin/env python3
"""Test ML model inference on board."""
import sys
sys.path.insert(0, '/opt/dms')
import numpy as np
from src.utils.npu import NPUModel

print('=== Testing ML Models on i.MX93 ===')
print()

# Test face detection model
print('Loading face_detection_ptq.tflite (CPU)...')
try:
    m = NPUModel('/opt/dms/models/face_detection_ptq.tflite', use_npu=False)
    inp = np.zeros((1, 128, 128, 3), dtype=np.float32)
    inp = (inp - 128.0) / 128.0
    out = m.predict(inp)
    print('  OK - Input:', m.input_shape, 'Outputs:', len(out))
    for i, o in enumerate(out):
        print('  Output %d:' % i, o.shape, o.dtype)
except Exception as e:
    print('  FAILED:', e)
    import traceback
    traceback.print_exc()

print()

# Test face landmark model
print('Loading face_landmark_ptq.tflite (CPU)...')
try:
    m = NPUModel('/opt/dms/models/face_landmark_ptq.tflite', use_npu=False)
    inp = np.zeros((1, 192, 192, 3), dtype=np.float32)
    inp = (inp - 128.0) / 128.0
    out = m.predict(inp)
    print('  OK - Input:', m.input_shape, 'Outputs:', len(out))
    for i, o in enumerate(out):
        print('  Output %d:' % i, o.shape, o.dtype)
except Exception as e:
    print('  FAILED:', e)
    import traceback
    traceback.print_exc()

print()

# Test iris model
print('Loading iris_landmark_ptq.tflite (CPU)...')
try:
    m = NPUModel('/opt/dms/models/iris_landmark_ptq.tflite', use_npu=False)
    # Try common input sizes
    for size in [(71, 71), (64, 64), (48, 48)]:
        try:
            inp = np.zeros((1, size[1], size[0], 3), dtype=np.float32)
            inp = (inp - 128.0) / 128.0
            out = m.predict(inp)
            print('  OK (%dx%d) - Input:' % (size[0], size[1]), m.input_shape, 'Outputs:', len(out))
            for i, o in enumerate(out):
                print('  Output %d:' % i, o.shape, o.dtype)
            break
        except Exception as e2:
            print('  %dx%d failed: %s' % (size[0], size[1], str(e2)[:80]))
except Exception as e:
    print('  FAILED:', e)
    import traceback
    traceback.print_exc()

print()
print('=== Model Test Complete ===')
