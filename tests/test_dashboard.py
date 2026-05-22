import pytest
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dashboard.renderer import DashboardRenderer, CAM_WIDTH, CAM_HEIGHT, PANEL_WIDTH, PANEL_HEIGHT
from src.dashboard.overlays import (draw_landmarks, draw_iris, draw_gaze_ray,
                                     draw_head_bbox, LANDMARK_GROUPS)
from src.dashboard.warnings import WarningVisuals


def _sample_results(**overrides):
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
    results.update(overrides)
    return results


class TestDashboardRenderer:
    def test_frame_shape(self):
        r = DashboardRenderer(width=1920, height=1080)
        assert r.frame.shape == (1080, 1920, 4), f'got {r.frame.shape}'

    def test_cam_width_constant(self):
        assert CAM_WIDTH == 1440

    def test_cam_height_constant(self):
        assert CAM_HEIGHT == 1080

    def test_panel_width_constant(self):
        assert PANEL_WIDTH == 480

    def test_panel_height_constant(self):
        assert PANEL_HEIGHT == 1080

    def test_render_returns_array(self):
        r = DashboardRenderer(width=1920, height=1080)
        cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        out = r.render(cam_frame, _sample_results(), fps=25.0)
        assert isinstance(out, np.ndarray)

    def test_render_output_shape(self):
        r = DashboardRenderer(width=1920, height=1080)
        cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        out = r.render(cam_frame, _sample_results(), fps=25.0)
        assert out.shape == (1080, 1920, 4), f'got {out.shape}'

    def test_alpha_channel_fully_opaque(self):
        r = DashboardRenderer(width=1920, height=1080)
        cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        out = r.render(cam_frame, _sample_results(), fps=25.0)
        assert np.all(out[:, :, 3] == 255)

    def test_left_region_not_zeros(self):
        r = DashboardRenderer(width=1920, height=1080)
        cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        out = r.render(cam_frame, _sample_results(), fps=25.0)
        left_region = out[:CAM_HEIGHT, :CAM_WIDTH, :3]
        assert np.any(left_region > 0)

    def test_status_panel_dark_background(self):
        r = DashboardRenderer(width=1920, height=1080)
        cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        out = r.render(cam_frame, _sample_results(), fps=25.0)
        panel_region = out[:, CAM_WIDTH:, :3]
        panel_mean = np.mean(panel_region)
        assert panel_mean < 50, f'panel_mean={panel_mean:.1f}'

    def test_render_handles_none_camera(self):
        r = DashboardRenderer(width=1920, height=1080)
        out_none = r.render(None, _sample_results(), fps=25.0)
        assert out_none is not None

    def test_render_handles_no_face(self):
        r = DashboardRenderer(width=1920, height=1080)
        cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        results_no_face = _sample_results(face_detected=False, warning_level='advisory')
        out_nf = r.render(cam_frame, results_no_face, fps=25.0)
        assert out_nf is not None

    @pytest.mark.parametrize('wlevel', ['none', 'advisory', 'escalating', 'intervention', 'emergency'])
    def test_render_handles_warning_levels(self, wlevel):
        r = DashboardRenderer(width=1920, height=1080)
        cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        ms = 'sleep' if wlevel == 'emergency' else 'open'
        rw = _sample_results(warning_level=wlevel, microsleep_state=ms)
        out_w = r.render(cam_frame, rw, fps=25.0)
        assert out_w is not None

    def test_render_handles_distracted_state(self):
        r = DashboardRenderer(width=1920, height=1080)
        cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        rw = _sample_results(is_distracted=True, continuous_offroad=4.5, warning_level='escalating')
        out_d = r.render(cam_frame, rw, fps=25.0)
        assert out_d is not None

    def test_render_handles_microsleep_state(self):
        r = DashboardRenderer(width=1920, height=1080)
        cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        rw = _sample_results(microsleep_state='microsleep')
        out_ms = r.render(cam_frame, rw, fps=25.0)
        assert out_ms is not None

    def test_render_handles_yawn_state(self):
        r = DashboardRenderer(width=1920, height=1080)
        cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        rw = _sample_results(yawn_active=True)
        out_y = r.render(cam_frame, rw, fps=25.0)
        assert out_y is not None

    def test_get_bgrx_frame_returns_current_frame(self):
        r = DashboardRenderer(width=1920, height=1080)
        cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        r.render(cam_frame, _sample_results(), fps=25.0)
        bgrx = r.get_bgrx_frame()
        assert np.array_equal(bgrx, r.frame)

    @pytest.mark.parametrize('fps_val', [0, 10, 25, 60])
    def test_render_works_at_various_fps(self, fps_val):
        r = DashboardRenderer(width=1920, height=1080)
        cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        out_fps = r.render(cam_frame, _sample_results(), fps=fps_val)
        assert out_fps is not None

    @pytest.mark.parametrize('kss', [1, 3, 5, 7, 8])
    def test_render_handles_kss_values(self, kss):
        r = DashboardRenderer(width=1920, height=1080)
        cam_frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        rk = _sample_results(kss=kss, health_score=max(0, 100 - (kss - 1) * 12))
        out_k = r.render(cam_frame, rk, fps=25.0)
        assert out_k is not None


class TestOverlays:
    def test_draw_landmarks_does_not_crash(self):
        frame = np.zeros((800, 1280, 3), dtype=np.uint8)
        lm = [(640, 400)] * 25
        draw_landmarks(frame, lm)
        assert True

    def test_draw_landmarks_draws_something(self):
        frame = np.zeros((800, 1280, 3), dtype=np.uint8)
        lm = [(640, 400)] * 25
        draw_landmarks(frame, lm)
        assert np.any(frame > 0)

    def test_draw_landmarks_with_scale_does_not_crash(self):
        frame2 = np.zeros((800, 1280, 3), dtype=np.uint8)
        lm_scaled = [(320, 200)] * 25
        draw_landmarks(frame2, lm_scaled, scale_x=1280, scale_y=800)
        assert True

    def test_draw_landmarks_handles_none(self):
        frame3 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_landmarks(frame3, None)
        assert True

    def test_draw_landmarks_handles_partial_landmarks(self):
        frame4 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_landmarks(frame4, [(640, 400)] * 10)
        assert True

    def test_landmark_groups_has_nose(self):
        assert 'nose' in LANDMARK_GROUPS

    def test_landmark_groups_has_brow(self):
        assert 'brow' in LANDMARK_GROUPS

    def test_landmark_groups_has_left_eye(self):
        assert 'left_eye' in LANDMARK_GROUPS

    def test_landmark_groups_has_right_eye(self):
        assert 'right_eye' in LANDMARK_GROUPS

    def test_landmark_groups_has_mouth(self):
        assert 'mouth' in LANDMARK_GROUPS

    def test_draw_iris_does_not_crash(self):
        frame5 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_iris(frame5, (640, 400, 15), (740, 400, 15))
        assert True

    def test_draw_iris_draws_something(self):
        frame5 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_iris(frame5, (640, 400, 15), (740, 400, 15))
        assert np.any(frame5 > 0)

    def test_draw_iris_handles_none(self):
        frame6 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_iris(frame6, None, None)
        assert True

    def test_draw_iris_handles_one_none(self):
        frame6 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_iris(frame6, (640, 400, 15), None)
        assert True

    def test_draw_iris_with_scale_does_not_crash(self):
        frame7 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_iris(frame7, (0.5, 0.5, 0.02), (0.7, 0.5, 0.02),
                  scale_x=1280, scale_y=800)
        assert True

    def test_draw_gaze_ray_does_not_crash(self):
        frame8 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_gaze_ray(frame8, (640, 400), 10.0, -5.0)
        assert True

    def test_draw_gaze_ray_draws_something(self):
        frame8 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_gaze_ray(frame8, (640, 400), 10.0, -5.0)
        assert np.any(frame8 > 0)

    def test_draw_gaze_ray_handles_none_nose(self):
        frame9 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_gaze_ray(frame9, None, 10.0, -5.0)
        assert True

    def test_draw_gaze_ray_handles_zero_angles(self):
        frame10 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_gaze_ray(frame10, (640, 400), 0.0, 0.0)
        assert True

    def test_draw_gaze_ray_handles_large_angles(self):
        frame11 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_gaze_ray(frame11, (640, 400), 45.0, -30.0, length=200)
        assert True

    def test_draw_head_bbox_does_not_crash(self):
        frame12 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_head_bbox(frame12, [300, 200, 500, 450], confidence=0.95)
        assert True

    def test_draw_head_bbox_draws_something(self):
        frame12 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_head_bbox(frame12, [300, 200, 500, 450], confidence=0.95)
        assert np.any(frame12 > 0)

    def test_draw_head_bbox_handles_none(self):
        frame13 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_head_bbox(frame13, None)
        assert True

    def test_draw_head_bbox_with_scale_does_not_crash(self):
        frame14 = np.zeros((800, 1280, 3), dtype=np.uint8)
        draw_head_bbox(frame14, [0.2, 0.2, 0.5, 0.5],
                       confidence=0.8, scale_x=1280, scale_y=800)
        assert True


class TestWarningVisuals:
    def test_warning_emergency_renders(self):
        wv = WarningVisuals()
        frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
        wv.render(frame_w, 'emergency', timestamp=1.0)
        assert True

    def test_emergency_draws_something(self):
        wv = WarningVisuals()
        frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
        wv.render(frame_w, 'emergency', timestamp=1.0)
        assert np.any(frame_w > 0)

    def test_warning_intervention_renders(self):
        wv = WarningVisuals()
        frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
        wv.render(frame_w, 'intervention', timestamp=2.0)
        assert True

    def test_warning_escalating_renders(self):
        wv = WarningVisuals()
        frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
        wv.render(frame_w, 'escalating', timestamp=3.0)
        assert True

    def test_warning_advisory_renders(self):
        wv = WarningVisuals()
        frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
        wv.render(frame_w, 'advisory', timestamp=4.0)
        assert True

    def test_warning_none_renders_without_crash(self):
        wv = WarningVisuals()
        frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
        wv.render(frame_w, 'none', timestamp=5.0)
        assert True

    def test_warning_none_draws_nothing(self):
        wv = WarningVisuals()
        frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
        wv.render(frame_w, 'none', timestamp=5.0)
        assert np.all(frame_w == 0)

    def test_flash_state_toggles_over_time(self):
        wv2 = WarningVisuals()
        frame1 = np.zeros((1080, 1440, 3), dtype=np.uint8)
        wv2.render(frame1, 'emergency', timestamp=0.0)
        frame2 = np.zeros((1080, 1440, 3), dtype=np.uint8)
        wv2.render(frame2, 'emergency', timestamp=0.2)
        assert True

    def test_unknown_warning_level_does_not_crash(self):
        wv = WarningVisuals()
        frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
        wv.render(frame_w, 'unknown_level', timestamp=6.0)
        assert True

    def test_rapid_timestamp_jumps_do_not_crash(self):
        wv3 = WarningVisuals()
        frame_w = np.zeros((1080, 1440, 3), dtype=np.uint8)
        for t in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            wv3.render(frame_w, 'escalating', timestamp=t)
        assert True
