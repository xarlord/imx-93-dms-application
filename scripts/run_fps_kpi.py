#!/usr/bin/env python3
"""DMS FPS KPI benchmark - camera + inference, no display."""
import sys
import time
import os

sys.path.insert(0, '/opt/dms')

from src.utils.config import DMSConfig
from src.pipeline import DMSPipeline
from src.capture import CameraCapture
import numpy as np


def main():
    cfg = DMSConfig('/opt/dms/config/config.yaml')

    test_modes = [
        ('CPU (no NPU)', False, False),
        ('NPU (Ethos-U65)', False, True),
    ]

    for mode_name, mock, use_npu in test_modes:
        cfg.set('models.mock', mock)
        cfg.set('models.use_npu', use_npu)
        if mock:
            cfg.set('models.head_detector', '')
            cfg.set('models.landmark', '')
            cfg.set('models.iris', '')

        print()
        print('=' * 60)
        print(f'  Mode: {mode_name}')
        print('=' * 60)

        t_build_start = time.monotonic()
        pipe = DMSPipeline(cfg)
        t_build = time.monotonic() - t_build_start
        print(f'Pipeline built in {t_build:.2f}s')

        cam_cfg = cfg.camera
        capture = CameraCapture(config=cam_cfg)

        frame_count = 0
        t0 = 0
        inference_times = []
        face_count = 0
        kss_samples = []

        def on_frame(frame_rgb):
            nonlocal frame_count, t0, face_count
            frame_count += 1
            t_proc_start = time.monotonic()
            results = pipe.process_frame(frame_rgb)
            t_proc = time.monotonic() - t_proc_start
            inference_times.append(t_proc)

            if results.get('face_detected'):
                face_count += 1
            kss_samples.append(results.get('kss', 0))

            if frame_count % 25 == 0:
                elapsed = time.monotonic() - t0
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                inf_fps = 1.0 / t_proc if t_proc > 0 else 0
                print(f'  Frame {frame_count:4d}: '
                      f'throughput={avg_fps:.1f} FPS, '
                      f'inference={inf_fps:.1f} FPS, '
                      f'KSS={results.get("kss", 0)} '
                      f'face={results.get("face_detected", False)} '
                      f'zone={results.get("zone_name", "?")} '
                      f'EAR={results.get("ear", 0):.3f}')

        capture.start(on_frame=on_frame)
        t0 = time.monotonic()
        print(f'Camera started. Capturing for 20s...')
        time.sleep(20)
        capture.stop()
        pipe.stop()

        elapsed = time.monotonic() - t0
        avg_fps = frame_count / elapsed if elapsed > 0 else 0

        if inference_times:
            sorted_t = sorted(inference_times)
            p50_ms = sorted_t[len(sorted_t) // 2] * 1000
            p95_ms = sorted_t[int(len(sorted_t) * 0.95)] * 1000
            min_ms = sorted_t[0] * 1000
            max_ms = sorted_t[-1] * 1000
            avg_inf_ms = sum(inference_times) / len(inference_times) * 1000
            avg_inf_fps = 1000.0 / avg_inf_ms if avg_inf_ms > 0 else 0
        else:
            p50_ms = p95_ms = min_ms = max_ms = avg_inf_ms = avg_inf_fps = 0

        face_rate = face_count / frame_count * 100 if frame_count > 0 else 0
        avg_kss = sum(kss_samples) / len(kss_samples) if kss_samples else 0

        print()
        print(f'  --- KPI Summary ({mode_name}) ---')
        print(f'  Total frames:        {frame_count}')
        print(f'  Elapsed time:        {elapsed:.1f}s')
        print(f'  Throughput FPS:      {avg_fps:.1f} (camera -> inference)')
        print(f'  Avg inference:       {avg_inf_fps:.1f} FPS ({avg_inf_ms:.1f} ms)')
        print(f'  P50 inference:       {1000.0/p50_ms:.1f} FPS ({p50_ms:.1f} ms)')
        print(f'  P95 inference:       {1000.0/p95_ms:.1f} FPS ({p95_ms:.1f} ms)')
        print(f'  Min inference:       {1000.0/max_ms:.1f} FPS ({max_ms:.1f} ms)')
        print(f'  Face detection rate: {face_rate:.1f}% ({face_count}/{frame_count})')
        print(f'  Avg KSS:             {avg_kss:.1f}')
        print(f'  Pipeline build:      {t_build:.2f}s')
        print(f'  Target:              25 FPS')
        print(f'  Status:              {"PASS" if avg_inf_fps >= 25 else "BELOW TARGET"} '
              f'({avg_inf_fps:.1f}/25 FPS)')


if __name__ == '__main__':
    main()
