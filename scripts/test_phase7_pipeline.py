"""Phase 7 tests: Pipeline orchestrator and main app (mock mode)."""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import DMSConfig
from src.pipeline import DMSPipeline

PASS = 0
FAIL = 0


def test(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  PASS: {name}')
    else:
        FAIL += 1
        print(f'  FAIL: {name} -- {detail}')


def main():
    global PASS, FAIL

    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = DMSConfig(config_path if os.path.exists(config_path) else None)

    config.set('models.mock', True)
    config.set('models.head_detector', '')
    config.set('models.landmark', '')
    config.set('models.iris', '')

    # ============================================================
    print('\n=== Phase 7A: DMSPipeline Construction ===')
    # ============================================================

    pipe = DMSPipeline(config)
    test('pipeline constructs', pipe is not None)
    test('pipeline has head detector', pipe.head_det is not None)
    test('pipeline has landmark detector', pipe.landmark_det is not None)
    test('pipeline has iris detector', pipe.iris_det is not None)
    test('pipeline has gaze estimator', pipe.gaze_est is not None)
    test('pipeline has PERCLOS', pipe.perclos is not None)
    test('pipeline has blink detector', pipe.blink is not None)
    test('pipeline has yawn detector', pipe.yawn is not None)
    test('pipeline has microsleep detector', pipe.microsleep is not None)
    test('pipeline has distraction detector', pipe.distraction is not None)
    test('pipeline has drowsiness scorer', pipe.drowsiness is not None)
    test('pipeline has warning manager', pipe.warning is not None)
    test('pipeline has renderer', pipe.renderer is not None)
    test('pipeline has warning visuals', pipe.warning_vis is not None)
    test('pipeline current_fps is 0 initially', pipe.current_fps == 0)

    # ============================================================
    print('\n=== Phase 7B: Single Frame Processing (Mock) ===')
    # ============================================================

    frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
    results = pipe.process_frame(frame)

    test('process_frame returns dict', isinstance(results, dict))

    required_keys = [
        'face_detected', 'head_bbox', 'landmarks', 'left_iris', 'right_iris',
        'ear_left', 'ear_right', 'ear', 'mar', 'kss', 'health_score',
        'perclos', 'perclos_severity', 'blink_rate', 'yawn_active',
        'microsleep_state', 'gaze_yaw', 'gaze_pitch', 'zone_id', 'zone_name',
        'is_on_road', 'is_distracted', 'warning_level', 'final_yaw',
        'final_pitch', 'proj_x', 'proj_y', 'continuous_offroad',
    ]
    for key in required_keys:
        test(f'results has key "{key}"', key in results,
             f'missing key, have: {sorted(results.keys())}')

    test('face detected in mock mode', results['face_detected'])

    if results['head_bbox']:
        x1, y1, x2, y2 = results['head_bbox']
        test('head bbox valid (x2>x1, y2>y1)', x2 > x1 and y2 > y1,
             f'({x1},{y1})-({x2},{y2})')

    test('25 landmarks detected',
         results['landmarks'] is not None and len(results['landmarks']) == 25,
         f'got {len(results["landmarks"]) if results["landmarks"] else 0}')

    test('left iris detected', results['left_iris'] is not None)
    test('right iris detected', results['right_iris'] is not None)

    test('EAR left in range', 0 <= results['ear_left'] <= 1.0,
         f'ear_left={results["ear_left"]}')
    test('EAR right in range', 0 <= results['ear_right'] <= 1.0,
         f'ear_right={results["ear_right"]}')

    test('MAR >= 0', results['mar'] >= 0, f'mar={results["mar"]}')

    test('KSS in range 1-9', 1 <= results['kss'] <= 9,
         f'kss={results["kss"]}')

    test('health_score in range', 0 <= results['health_score'] <= 100,
         f'health={results["health_score"]}')

    test('PERCLOS in range', 0 <= results['perclos'] <= 1.0,
         f'perclos={results["perclos"]}')

    test('blink_rate >= 0', results['blink_rate'] >= 0)

    valid_warnings = {'none', 'advisory', 'escalating', 'intervention', 'emergency'}
    test('warning level valid', results['warning_level'] in valid_warnings,
         f'got "{results["warning_level"]}"')

    valid_ms = {'open', 'closed', 'microsleep', 'sleep', 'unresponsive'}
    test('microsleep state valid', results['microsleep_state'] in valid_ms,
         f'got "{results["microsleep_state"]}"')

    # ============================================================
    print('\n=== Phase 7C: Dashboard Rendering ===')
    # ============================================================

    dashboard = pipe.render_dashboard(frame, results, fps=25.0)
    test('render_dashboard returns array', isinstance(dashboard, np.ndarray))
    test('dashboard shape is 1080x1920x4',
         dashboard.shape == (1080, 1920, 4),
         f'got {dashboard.shape}')
    test('dashboard dtype is uint8', dashboard.dtype == np.uint8)

    # ============================================================
    print('\n=== Phase 7D: Multi-Frame Consistency ===')
    # ============================================================

    all_ok = True
    for i in range(10):
        f = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        r = pipe.process_frame(f)
        if r['kss'] < 1 or r['kss'] > 9:
            all_ok = False
        if r['warning_level'] not in valid_warnings:
            all_ok = False

    test('10 consecutive frames process correctly', all_ok)

    test('FPS updates after processing',
         pipe.current_fps > 0 or pipe._frame_times,
         f'fps={pipe.current_fps}')

    # ============================================================
    print('\n=== Phase 7E: Behavioral State Changes ===')
    # ============================================================

    pipe2 = DMSPipeline(config)

    for i in range(30):
        f = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        pipe2.process_frame(f)

    test('pipeline handles 30 frames without crash', True)

    test('PERCLOS state is tracked', pipe2.perclos is not None)
    test('Blink detector tracks blinks', pipe2.blink.blink_count >= 0)
    test('Drowsiness scorer tracks health', 0 <= pipe2.drowsiness.health_score <= 100)

    # ============================================================
    print('\n=== Phase 7F: Edge Cases ===')
    # ============================================================

    pipe3 = DMSPipeline(config)
    empty = np.zeros((800, 1280, 3), dtype=np.uint8)
    r_empty = pipe3.process_frame(empty)
    test('pipeline handles black frame', r_empty is not None)

    small = np.random.randint(0, 255, (100, 160, 3), dtype=np.uint8)
    r_small = pipe3.process_frame(small)
    test('pipeline handles small frame', r_small is not None)

    large = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    r_large = pipe3.process_frame(large)
    test('pipeline handles large frame', r_large is not None)

    r_no_face = dict(r_empty)
    r_no_face['face_detected'] = False
    r_no_face['landmarks'] = None
    d_no_face = pipe3.render_dashboard(empty, r_no_face, fps=25.0)
    test('dashboard renders with no face', d_no_face.shape == (1080, 1920, 4))

    # ============================================================
    print(f'\n{"="*60}')
    print(f'Phase 7 Pipeline Results: {PASS} PASS, {FAIL} FAIL, {PASS+FAIL} total')
    if FAIL > 0:
        print('SOME TESTS FAILED')
    else:
        print('ALL TESTS PASSED')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
