import pytest
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import DMSConfig
from src.pipeline import DMSPipeline


@pytest.fixture()
def mock_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = DMSConfig(config_path if os.path.exists(config_path) else None)
    config.set('models.mock', True)
    config.set('models.head_detector', '')
    config.set('models.landmark', '')
    config.set('models.iris', '')
    return config


class TestDMSPipeline:
    def test_pipeline_constructs(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe is not None

    def test_pipeline_has_head_detector(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.head_det is not None

    def test_pipeline_has_landmark_detector(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.landmark_det is not None

    def test_pipeline_has_iris_detector(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.iris_det is not None

    def test_pipeline_has_gaze_estimator(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.gaze_est is not None

    def test_pipeline_has_perclos(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.perclos is not None

    def test_pipeline_has_blink_detector(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.blink is not None

    def test_pipeline_has_yawn_detector(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.yawn is not None

    def test_pipeline_has_microsleep_detector(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.microsleep is not None

    def test_pipeline_has_distraction_detector(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.distraction is not None

    def test_pipeline_has_drowsiness_scorer(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.drowsiness is not None

    def test_pipeline_has_warning_manager(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.warning is not None

    def test_pipeline_has_renderer(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.renderer is not None

    def test_pipeline_has_warning_visuals(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.warning_vis is not None

    def test_pipeline_current_fps_zero_initially(self, mock_config):
        pipe = DMSPipeline(mock_config)
        assert pipe.current_fps == 0

    def test_process_frame_returns_dict(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        assert isinstance(results, dict)

    @pytest.mark.parametrize('key', [
        'face_detected', 'head_bbox', 'landmarks', 'left_iris', 'right_iris',
        'ear_left', 'ear_right', 'ear', 'mar', 'kss', 'health_score',
        'perclos', 'perclos_severity', 'blink_rate', 'yawn_active',
        'microsleep_state', 'gaze_yaw', 'gaze_pitch', 'zone_id', 'zone_name',
        'is_on_road', 'is_distracted', 'warning_level', 'final_yaw',
        'final_pitch', 'proj_x', 'proj_y', 'continuous_offroad',
    ])
    def test_results_has_key(self, mock_config, key):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        assert key in results, f'missing key, have: {sorted(results.keys())}'

    def test_face_detected_in_mock_mode(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        assert results['face_detected']

    def test_head_bbox_valid(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        if results['head_bbox']:
            x1, y1, x2, y2 = results['head_bbox']
            assert x2 > x1 and y2 > y1, f'({x1},{y1})-({x2},{y2})'

    def test_25_landmarks_detected(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        assert results['landmarks'] is not None and len(results['landmarks']) == 25, \
            f'got {len(results["landmarks"]) if results["landmarks"] else 0}'

    def test_left_iris_detected(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        assert results['left_iris'] is not None

    def test_right_iris_detected(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        assert results['right_iris'] is not None

    def test_ear_left_in_range(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        assert 0 <= results['ear_left'] <= 1.0, f'ear_left={results["ear_left"]}'

    def test_ear_right_in_range(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        assert 0 <= results['ear_right'] <= 1.0, f'ear_right={results["ear_right"]}'

    def test_mar_non_negative(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        assert results['mar'] >= 0, f'mar={results["mar"]}'

    def test_kss_in_range(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        assert 1 <= results['kss'] <= 9, f'kss={results["kss"]}'

    def test_health_score_in_range(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        assert 0 <= results['health_score'] <= 100, f'health={results["health_score"]}'

    def test_perclos_in_range(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        assert 0 <= results['perclos'] <= 1.0, f'perclos={results["perclos"]}'

    def test_blink_rate_non_negative(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        assert results['blink_rate'] >= 0

    def test_warning_level_valid(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        valid_warnings = {'none', 'advisory', 'escalating', 'intervention', 'emergency'}
        assert results['warning_level'] in valid_warnings, f'got "{results["warning_level"]}"'

    def test_microsleep_state_valid(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        valid_ms = {'open', 'closed', 'microsleep', 'sleep', 'unresponsive'}
        assert results['microsleep_state'] in valid_ms, f'got "{results["microsleep_state"]}"'

    def test_render_dashboard_returns_array(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        dashboard = pipe.render_dashboard(frame, results, fps=25.0)
        assert isinstance(dashboard, np.ndarray)

    def test_dashboard_shape(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        dashboard = pipe.render_dashboard(frame, results, fps=25.0)
        assert dashboard.shape == (1080, 1920, 4), f'got {dashboard.shape}'

    def test_dashboard_dtype_uint8(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        results = pipe.process_frame(frame)
        dashboard = pipe.render_dashboard(frame, results, fps=25.0)
        assert dashboard.dtype == np.uint8

    def test_10_consecutive_frames_process_correctly(self, mock_config):
        pipe = DMSPipeline(mock_config)
        valid_warnings = {'none', 'advisory', 'escalating', 'intervention', 'emergency'}
        all_ok = True
        for i in range(10):
            f = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
            r = pipe.process_frame(f)
            if r['kss'] < 1 or r['kss'] > 9:
                all_ok = False
            if r['warning_level'] not in valid_warnings:
                all_ok = False
        assert all_ok

    def test_fps_updates_after_processing(self, mock_config):
        pipe = DMSPipeline(mock_config)
        frame = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
        for i in range(2):
            pipe.process_frame(frame)
        assert pipe.current_fps > 0 or pipe._frame_times, f'fps={pipe.current_fps}'

    def test_pipeline_handles_30_frames(self, mock_config):
        pipe2 = DMSPipeline(mock_config)
        for i in range(30):
            f = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
            pipe2.process_frame(f)
        assert True

    def test_perclos_state_is_tracked(self, mock_config):
        pipe2 = DMSPipeline(mock_config)
        for i in range(30):
            f = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
            pipe2.process_frame(f)
        assert pipe2.perclos is not None

    def test_blink_detector_tracks_blinks(self, mock_config):
        pipe2 = DMSPipeline(mock_config)
        for i in range(30):
            f = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
            pipe2.process_frame(f)
        assert pipe2.blink.blink_count >= 0

    def test_drowsiness_scorer_tracks_health(self, mock_config):
        pipe2 = DMSPipeline(mock_config)
        for i in range(30):
            f = np.random.randint(50, 200, (800, 1280, 3), dtype=np.uint8)
            pipe2.process_frame(f)
        assert 0 <= pipe2.drowsiness.health_score <= 100

    def test_pipeline_handles_black_frame(self, mock_config):
        pipe3 = DMSPipeline(mock_config)
        empty = np.zeros((800, 1280, 3), dtype=np.uint8)
        r_empty = pipe3.process_frame(empty)
        assert r_empty is not None

    def test_pipeline_handles_small_frame(self, mock_config):
        pipe3 = DMSPipeline(mock_config)
        small = np.random.randint(0, 255, (100, 160, 3), dtype=np.uint8)
        r_small = pipe3.process_frame(small)
        assert r_small is not None

    def test_pipeline_handles_large_frame(self, mock_config):
        pipe3 = DMSPipeline(mock_config)
        large = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
        r_large = pipe3.process_frame(large)
        assert r_large is not None

    def test_dashboard_renders_with_no_face(self, mock_config):
        pipe3 = DMSPipeline(mock_config)
        empty = np.zeros((800, 1280, 3), dtype=np.uint8)
        r_empty = pipe3.process_frame(empty)
        r_no_face = dict(r_empty)
        r_no_face['face_detected'] = False
        r_no_face['landmarks'] = None
        d_no_face = pipe3.render_dashboard(empty, r_no_face, fps=25.0)
        assert d_no_face.shape == (1080, 1920, 4)
