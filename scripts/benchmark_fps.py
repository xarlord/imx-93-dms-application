#!/usr/bin/env python3
"""Quick FPS benchmark for DMS pipeline on FRDM-IMX93."""
import sys
import time

sys.path.insert(0, '/opt/dms')

from src.utils.config import DMSConfig
from src.pipeline import DMSPipeline
import numpy as np


def benchmark_mock(n_frames=50):
    cfg = DMSConfig('/opt/dms/config/config.yaml')
    cfg.set('models.mock', True)
    cfg.set('models.head_detector', '')
    cfg.set('models.landmark', '')
    cfg.set('models.iris', '')
    cfg.set('models.use_npu', False)

    print('Building pipeline (mock, CPU)...')
    t0 = time.monotonic()
    pipe = DMSPipeline(cfg)
    t_build = time.monotonic() - t0
    print(f'Pipeline built in {t_build:.2f}s')

    print(f'Running {n_frames} mock frames...')
    t0 = time.monotonic()
    for i in range(n_frames):
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
    elapsed = time.monotonic() - t0
    fps = n_frames / elapsed
    print(f'{n_frames} frames in {elapsed:.2f}s = {fps:.1f} FPS (mock, CPU)')
    print(f'Last frame: KSS={results["kss"]}, warning={results["warning_level"]}, '
          f'face={results["face_detected"]}, zone={results["zone_name"]}')
    pipe.stop()


def benchmark_real(n_frames=30):
    cfg = DMSConfig('/opt/dms/config/config.yaml')
    cfg.set('models.mock', False)
    cfg.set('models.use_npu', False)

    print('Building pipeline (real models, CPU)...')
    t0 = time.monotonic()
    pipe = DMSPipeline(cfg)
    t_build = time.monotonic() - t0
    print(f'Pipeline built in {t_build:.2f}s')

    print(f'Running {n_frames} frames with real models...')
    t0 = time.monotonic()
    for i in range(n_frames):
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        if (i + 1) % 5 == 0:
            fps_so_far = (i + 1) / (time.monotonic() - t0)
            print(f'  Frame {i+1}/{n_frames}: {fps_so_far:.1f} FPS, '
                  f'KSS={results["kss"]}, face={results["face_detected"]}')
    elapsed = time.monotonic() - t0
    fps = n_frames / elapsed
    print(f'{n_frames} frames in {elapsed:.2f}s = {fps:.1f} FPS (real models, CPU)')
    print(f'Last frame: KSS={results["kss"]}, warning={results["warning_level"]}, '
          f'face={results["face_detected"]}, zone={results["zone_name"]}, '
          f'EAR={results["ear"]:.3f}, blink_rate={results["blink_rate"]:.1f}')
    pipe.stop()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['mock', 'real'], default='mock')
    parser.add_argument('--frames', type=int, default=50)
    args = parser.parse_args()

    print('=' * 60)
    print('  DMS FPS Benchmark - FRDM-IMX93')
    print('=' * 60)
    print()

    if args.mode == 'mock':
        benchmark_mock(args.frames)
    else:
        benchmark_real(args.frames)
