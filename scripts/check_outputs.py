#!/usr/bin/env python3
import tflite_runtime.interpreter as tflite

m = tflite.Interpreter(model_path='/opt/dms/models/face_detection_ptq.tflite')
m.allocate_tensors()
print('CPU model outputs:')
for i, d in enumerate(m.get_output_details()):
    print(f'  [{i}] name={d["name"]}, shape={d["shape"]}, dtype={d["dtype"]}')

print()

delegate = tflite.load_delegate('/usr/lib/libethosu_delegate.so')
m2 = tflite.Interpreter(model_path='/opt/gopoint-apps/downloads/face_detection_ptq_vela.tflite', experimental_delegates=[delegate])
m2.allocate_tensors()
print('NPU model outputs:')
for i, d in enumerate(m2.get_output_details()):
    print(f'  [{i}] name={d["name"]}, shape={d["shape"]}, dtype={d["dtype"]}')
