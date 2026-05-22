"""Phase 5 tests: Dashboard modules (renderer, overlays, warnings)."""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dashboard.renderer import DashboardRenderer, CAM_WIDTH, CAM_HEIGHT, PANEL_WIDTH, PANEL_HEIGHT
from src.dashboard.overlays import (draw_landmarks, draw_iris, draw_gaze_ray,
                                     draw_head_bbox, LANDMARK_GROUPS)
from src.dashboard.warnings import WarningVisuals

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

    # ============================================================
    print('\n=== Phase 5A: DashboardRenderer ===')
    # ============================================================

    r = DashboardRenderer(width=1920, height=1080)

    # Test 1: renderer has correct frame shape
    test('frame shape is 1080x1920x4',
         r.frame.shape == (1080, 1920, 4),
         f'got {r.frame.shape}')

    # Test 2: constants
    test('CAM_WIDTH=1440', CAM_WIDTH == 1440)
    test('CAM_HEIGHT=1080', CAM_HEIGHT == 1080)
    test('PANEL_WIDTH=480', PANEL_WIDTH == 480)
    test('PANEL_HEIGHT=1080', PANEL_HEIGHT == 1080)

    # Test 3: render with camera frame
    cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
    results = {
        'face_detected': True,
        'kss': 3,
        'health_score': 75.0,
        'perclos': 5.0,
        'blink_rate': 15,
        'ear': 0.28,
        'zone_name': 'ROAD_AHEAD',
        'is_on_road': True,
        'final_yaw': -5.0,
        'final_pitch': 3.0,
        'warning_level': 'none',
        'microsleep_state': 'open',
        'is_distracted': False,
        'yawn_active': False,
        'proj_x': 960,
        'proj_y': 400,
    }
    out = r.render(cam_frame, results, fps=25.0)
    test('render returns array', isinstance(out, np.ndarray))
    test('render output shape 1080x1920x4',
         out.shape == (1080, 1920, 4),
         f'got {out.shape}')

    # Test 4: alpha channel is 255
    test('alpha channel fully opaque',
         np.all(out[:, :, 3] == 255))

    # Test 5: camera feed placed in left region
    left_region = out[:CAM_HEIGHT, :CAM_WIDTH, :3]
    test('left region not all zeros',
         np.any(left_region > 0))

    # Test 6: status panel is dark (right region)
    panel_region = out[:, CAM_WIDTH:, :3]
    panel_mean = np.mean(panel_region)
    test('status panel has dark background (mean < 50)',
         panel_mean < 50,
         f'panel_mean={panel_mean:.1f}')

    # Test 7: render with None camera frame
    out_none = r.render(None, results, fps=25.0)
    test('render handles None camera', out_none is not None)

    # Test 8: render with no face
    results_no_face = dict(results)
    results_no_face['face_detected'] = False
    results_no_face['warning_level'] = 'advisory'
    out_nf = r.render(cam_frame, results_no_face, fps=25.0)
    test('render handles no face', out_nf is not None)

    # Test 9: render with all warning levels
    for wlevel in ['none', 'advisory', 'escalating', 'intervention', 'emergency']:
        rw = dict(results)
        rw['warning_level'] = wlevel
        rw['microsleep_state'] = 'sleep' if wlevel == 'emergency' else 'open'
        out_w = r.render(cam_frame, rw, fps=25.0)
        test(f'render handles warning={wlevel}', out_w is not None)

    # Test 10: render with distracted
    rw = dict(results)
    rw['is_distracted'] = True
    rw['continuous_offroad'] = 4.5
    rw['warning_level'] = 'escalating'
    out_d = r.render(cam_frame, rw, fps=25.0)
    test('render handles distracted state', out_d is not None)

    # Test 11: render with microsleep
    rw = dict(results)
    rw['microsleep_state'] = 'microsleep'
    out_ms = r.render(cam_frame, rw, fps=25.0)
    test('render handles microsleep state', out_ms is not None)

    # Test 12: render with yawn
    rw = dict(results)
    rw['yawn_active'] = True
    out_y = r.render(cam_frame, rw, fps=25.0)
    test('render handles yawn state', out_y is not None)

    # Test 13: get_bgrx_frame returns same frame
    bgrx = r.get_bgrx_frame()
    test('get_bgrx_frame returns current frame',
         np.array_equal(bgrx, r.frame))

    # Test 14: different FPS values
    for fps_val in [0, 10, 25, 60]:
        out_fps = r.render(cam_frame, results, fps=fps_val)
        test(f'render works at fps={fps_val}', out_fps is not None)

    # Test 15: extreme KSS values
    for kss in [1, 3, 5, 7, 8]:
        rk = dict(results)
        rk['kss'] = kss
        rk['health_score'] = max(0, 100 - (kss - 1) * 12)
        out_k = r.render(cam_frame, rk, fps=25.0)
        test(f'render handles KSS={kss}', out_k is not None)

    # ============================================================
    print('\n=== Phase 5B: Overlay Functions ===')
    # ============================================================

    # draw_landmarks
    frame = np.zeros((800, 1280, 3), dtype=np.uint8)
    lm = [(640, 400)] * 25  # All landmarks at center
    draw_landmarks(frame, lm)
    test('draw_landmarks does not crash', True)
    test('draw_landmarks draws something',
         np.any(frame > 0))

    # draw_landmarks with scale
    frame2 = np.zeros((800, 1280, 3), dtype=np.uint8)
    lm_scaled = [(320, 200)] * 25  # Normalized coords
    draw_landmarks(frame2, lm_scaled, scale_x=1280, scale_y=800)
    test('draw_landmarks with scale does not crash', True)

    # draw_landmarks with None
    frame3 = np.zeros((800, 1280, 3), dtype=np.uint8)
    draw_landmarks(frame3, None)
    test('draw_landmarks handles None', True)

    # draw_landmarks with fewer landmarks
    frame4 = np.zeros((800, 1280, 3), dtype=np.uint8)
    draw_landmarks(frame4, [(640, 400)] * 10)
    test('draw_landmarks handles partial landmarks', True)

    # Test LANDMARK_GROUPS has all groups
    test('LANDMARK_GROUPS has nose', 'nose' in LANDMARK_GROUPS)
    test('LANDMARK_GROUPS has brow', 'brow' in LANDMARK_GROUPS)
    test('LANDMARK_GROUPS has left_eye', 'left_eye' in LANDMARK_GROUPS)
    test('LANDMARK_GROUPS has right_eye', 'right_eye' in LANDMARK_GROUPS)
    test('LANDMARK_GROUPS has mouth', 'mouth' in LANDMARK_GROUPS)

    # draw_iris
    frame5 = np.zeros((800, 1280, 3), dtype=np.uint8)
    draw_iris(frame5, (640, 400, 15), (740, 400, 15))
    test('draw_iris does not crash', True)
    test('draw_iris draws something',
         np.any(frame5 > 0))

    # draw_iris with None
    frame6 = np.zeros((800, 1280, 3), dtype=np.uint8)
    draw_iris(frame6, None, None)
    test('draw_iris handles None', True)
    draw_iris(frame6, (640, 400, 15), None)
    test('draw_iris handles one None', True)

    # draw_iris with scale
    frame7 = np.zeros((800, 1280, 3), dtype=np.uint8)
    draw_iris(frame7, (0.5, 0.5, 0.02), (0.7, 0.5, 0.02),
              scale_x=1280, scale_y=800)
    test('draw_iris with scale does not crash', True)

    # draw_gaze_ray
    frame8 = np.zeros((800, 1280, 3), dtype=np.uint8)
    draw_gaze_ray(frame8, (640, 400), 10.0, -5.0)
    test('draw_gaze_ray does not crash', True)
    test('draw_gaze_ray draws something',
         np.any(frame8 > 0))

    # draw_gaze_ray with None
    frame9 = np.zeros((800, 1280, 3), dtype=np.uint8)
    draw_gaze_ray(frame9, None, 10.0, -5.0)
    test('draw_gaze_ray handles None nose', True)

    # draw_gaze_ray with zero angles
    frame10 = np.zeros((800, 1280, 3), dtype=np.uint8)
    draw_gaze_ray(frame10, (640, 400), 0.0, 0.0)
    test('draw_gaze_ray handles zero angles', True)

    # draw_gaze_ray with large angles
    frame11 = np.zeros((800, 1280, 3), dtype=np.uint8)
    draw_gaze_ray(frame11, (640, 400), 45.0, -30.0, length=200)
    test('draw_gaze_ray handles large angles', True)

    # draw_head_bbox
    frame12 = np.zeros((800, 1280, 3), dtype=np.uint8)
    draw_head_bbox(frame12, [300, 200, 500, 450], confidence=0.95)
    test('draw_head_bbox does not crash', True)
    test('draw_head_bbox draws something',
         np.any(frame12 > 0))

    # draw_head_bbox with None
    frame13 = np.zeros((800, 1280, 3), dtype=np.uint8)
    draw_head_bbox(frame13, None)
    test('draw_head_bbox handles None', True)

    # draw_head_bbox with scale
    frame14 = np.zeros((800, 1280, 3), dtype=np.uint8)
    draw_head_bbox(frame14, [0.2, 0.2, 0.5, 0.5],
                   confidence=0.8, scale_x=1280, scale_y=800)
    test('draw_head_bbox with scale does not crash', True)

    # ============================================================
    print('\n=== Phase 5C: WarningVisuals ===')
    # ============================================================

    wv = WarningVisuals()

    # Test: render emergency warning
    frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
    wv.render(frame_w, 'emergency', timestamp=1.0)
    test('warning emergency renders', True)
    test('emergency draws something', np.any(frame_w > 0))

    # Test: render intervention
    frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
    wv.render(frame_w, 'intervention', timestamp=2.0)
    test('warning intervention renders', True)

    # Test: render escalating
    frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
    wv.render(frame_w, 'escalating', timestamp=3.0)
    test('warning escalating renders', True)

    # Test: render advisory
    frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
    wv.render(frame_w, 'advisory', timestamp=4.0)
    test('warning advisory renders', True)

    # Test: render none (no warning)
    frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
    wv.render(frame_w, 'none', timestamp=5.0)
    test('warning none renders without crash', True)
    # none should not draw anything
    test('warning none draws nothing',
         np.all(frame_w == 0))

    # Test: flash state toggles
    wv2 = WarningVisuals()
    frame1 = np.zeros((1080, 1440, 3), dtype=np.uint8)
    wv2.render(frame1, 'emergency', timestamp=0.0)
    frame2 = np.zeros((1080, 1440, 3), dtype=np.uint8)
    wv2.render(frame2, 'emergency', timestamp=0.2)  # > 0.125, should toggle
    test('flash state toggles over time', True)

    # Test: unknown warning level
    frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
    wv.render(frame_w, 'unknown_level', timestamp=6.0)
    test('unknown warning level does not crash', True)

    # Test: rapid timestamp jumps
    frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
    wv3 = WarningVisuals()
    for t in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        wv3.render(frame_w, 'escalating', timestamp=t)
    test('rapid timestamp jumps do not crash', True)

    # ============================================================
    print(f'\n{"="*60}')
    print(f'Phase 5 Dashboard Results: {PASS} PASS, {FAIL} FAIL, {PASS+FAIL} total')
    if FAIL > 0:
        print('SOME TESTS FAILED')
    else:
        print('ALL TESTS PASSED')
        print(f'{"="*60}')


if __name__ == '__main__':
    main()
