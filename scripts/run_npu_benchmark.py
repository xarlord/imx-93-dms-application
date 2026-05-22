#!/usr/bin/env python3
"""DMS FPS KPI benchmark - synthetic frames + NPU, no camera, no display."""
import sys
import time

sys.path.insert(0, '/opt/dms')

import numpy as np
from src.utils.config import DMSConfig
from src.utils.npu import NPUModel

VELA_MODELS = '/opt/gopoint-apps/downloads'
DMS_MODELS = '/opt/dms/models'
DELEGATE_PATH = '/usr/lib/libethosu_delegate.so'


def load_npu_model(model_path: str) -> tuple:
    import tflite_runtime.interpreter as tflite
    delegate = tflite.load_delegate(DELEGATE_PATH)
    interp = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[delegate],
    )
    interp.allocate_tensors()
    return interp


def load_cpu_model(model_path: str) -> tuple:
    import tflite_runtime.interpreter as tflite
    interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def benchmark_model(interp, input_shape, input_dtype, name, n_warmup=3, n_runs=50):
    inp = np.zeros(input_shape, dtype=input_dtype)
    idx = interp.get_input_details()[0]['index']

    for _ in range(n_warmup):
        interp.set_tensor(idx, inp)
        interp.invoke()

    times = []
    for _ in range(n_runs):
        t0 = time.monotonic()
        interp.set_tensor(idx, inp)
        interp.invoke()
        times.append(time.monotonic() - t0)

    avg_ms = sum(times) / len(times) * 1000
    avg_fps = 1000.0 / avg_ms
    sorted_t = sorted(times)
    p50_ms = sorted_t[len(sorted_t) // 2] * 1000
    p95_ms = sorted_t[int(len(sorted_t) * 0.95)] * 1000
    min_ms = sorted_t[0] * 1000
    print(f'  {name}: avg={avg_fps:.1f} FPS ({avg_ms:.1f}ms), '
          f'p50={1000/p50_ms:.1f} FPS, p95={1000/p95_ms:.1f} FPS, '
          f'min={1000/min_ms:.1f} FPS')
    return avg_ms


def main():
    models_npu = [
        ('face_detection', f'{VELA_MODELS}/face_detection_ptq_vela.tflite', (1, 128, 128, 3), np.float32),
        ('face_landmark', f'{VELA_MODELS}/face_landmark_ptq_vela.tflite', (1, 192, 192, 3), np.float32),
        ('iris_landmark', f'{VELA_MODELS}/iris_landmark_ptq_vela.tflite', (1, 64, 64, 3), np.float32),
    ]
    models_cpu = [
        ('face_detection', f'{DMS_MODELS}/face_detection_ptq.tflite', (1, 128, 128, 3), np.float32),
        ('face_landmark', f'{DMS_MODELS}/face_landmark_ptq.tflite', (1, 192, 192, 3), np.float32),
        ('iris_landmark', f'{DMS_MODELS}/iris_landmark_ptq.tflite', (1, 64, 64, 3), np.float32),
    ]

    for mode, models in [('NPU (Ethos-U65)', models_npu), ('CPU (XNNPACK)', models_cpu)]:
        print()
        print('=' * 60)
        print(f'  Mode: {mode}')
        print('=' * 60)
        total_ms = 0
        for name, path, shape, dtype in models:
            try:
                if 'NPU' in mode:
                    interp = load_npu_model(path)
                else:
                    interp = load_cpu_model(path)
                inf_ms = benchmark_model(interp, shape, dtype, name)
                total_ms += inf_ms
                del interp
            except Exception as e:
                print(f'  {name}: FAILED - {e}')
        total_fps = 1000.0 / total_ms if total_ms > 0 else 0
        print(f'  --- Pipeline Total ---')
        print(f'  Sequential 3-model: {total_fps:.1f} FPS ({total_ms:.1f}ms)')
        print(f'  Target:              25 FPS')
        print(f'  Status:              {"PASS" if total_fps >= 25 else "BELOW TARGET"} '
              f'({total_fps:.1f}/25 FPS)')

    print()
    print('=' * 60)
    print('  DMS Pipeline Benchmark (NPU mode, full pipeline)')
    print('=' * 60)
    cfg = DMSConfig('/opt/dms/config/config.yaml')
    cfg.set('models.use_npu', True)
    cfg.set('models.head_detector', f'{VELA_MODELS}/face_detection_ptq_vela.tflite')
    cfg.set('models.landmark', f'{VELA_MODELS}/face_landmark_ptq_vela.tflite')
    cfg.set('models.iris', f'{VELA_MODELS}/iris_landmark_ptq_vela.tflite')

    from src.pipeline import DMSPipeline
    pipe = DMSPipeline(cfg)
    backends = []
    for attr_name in ['head_det', 'landmark_det', 'iris_det']:
        det = getattr(pipe, attr_name, None)
        if det and hasattr(det, 'model') and hasattr(det.model, 'backend'):
            backends.append(f'{attr_name}={det.model.backend}')
    print(f'Pipeline backends: {", ".join(backends)}')

    synthetic = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)

    for _ in range(3):
        pipe.process_frame(synthetic)

    times = []
    for i in range(50):
        t0 = time.monotonic()
        pipe.process_frame(synthetic)
        times.append(time.monotonic() - t0)

    avg_ms = sum(times) / len(times) * 1000
    avg_fps = 1000.0 / avg_ms
    sorted_t = sorted(times)
    p50_ms = sorted_t[len(sorted_t) // 2] * 1000
    p95_ms = sorted_t[int(len(sorted_t) * 0.95)] * 1000

    print(f'  Full pipeline: avg={avg_fps:.1f} FPS ({avg_ms:.1f}ms)')
    print(f'  P50={1000/p50_ms:.1f} FPS, P95={1000/p95_ms:.1f} FPS')
    print(f'  Target: 25 FPS')
    print(f'  Status: {"PASS" if avg_fps >= 25 else "BELOW TARGET"} ({avg_fps:.1f}/25 FPS)')

    pipe.stop()


if __name__ == '__main__':
    main()
