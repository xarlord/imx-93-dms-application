import pytest
import sys
import os
import math
import json
import time
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gaze_estimator import GazeEstimator
from src.calibration.gaze_calibration import GazeCalibration, CALIBRATION_POINTS
from src.calibration.baseline import BaselineLearner
from src.utils.zone_loader import ZoneLoader


def _make_landmarks():
    landmarks = [(640, 400)]
    landmarks += [(580, 360), (600, 360)]
    landmarks += [(680, 360), (700, 360)]
    landmarks += [(570, 370), (580, 362), (595, 362), (600, 370), (595, 378), (580, 378)]
    landmarks += [(710, 370), (700, 362), (685, 362), (680, 370), (685, 378), (700, 378)]
    landmarks += [(610, 430), (625, 425), (640, 424), (655, 425),
                  (670, 430), (655, 436), (640, 437), (625, 436)]
    assert len(landmarks) == 25
    return landmarks


def _make_gaze_estimator():
    zones_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'zones.xml')
    zl = ZoneLoader(zones_path)
    ge = GazeEstimator(yaw_weight=32.0, pitch_weight=45.0, ppd_yaw=5.0, ppd_pitch=5.0,
                       display_w=1920, display_h=1080)
    ge.set_zone_loader(zl)
    return ge


FACE_BBOX = [540, 275, 740, 525]


class TestGazeEstimator:
    def test_straight_ahead_small_yaw(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        result = ge.estimate(landmarks, (585, 370), (695, 370), FACE_BBOX)
        assert abs(result['final_yaw']) < 10, f"yaw={result['final_yaw']:.1f}"

    def test_straight_ahead_small_pitch(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        result = ge.estimate(landmarks, (585, 370), (695, 370), FACE_BBOX)
        assert abs(result['final_pitch']) < 10, f"pitch={result['final_pitch']:.1f}"

    def test_straight_ahead_road_ahead_zone(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        result = ge.estimate(landmarks, (585, 370), (695, 370), FACE_BBOX)
        assert result['zone_name'] == 'ROAD_AHEAD', f"zone={result['zone_name']}"

    def test_look_left_negative_yaw(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        ge.reset()
        result = ge.estimate(landmarks, (572, 370), (682, 370), FACE_BBOX)
        assert result['final_yaw'] < -3, f"yaw={result['final_yaw']:.1f}"

    def test_look_right_positive_yaw(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        ge.reset()
        result = ge.estimate(landmarks, (598, 370), (708, 370), FACE_BBOX)
        assert result['final_yaw'] > 3, f"yaw={result['final_yaw']:.1f}"

    def test_look_down_positive_pitch(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        ge.reset()
        result = ge.estimate(landmarks, (585, 377), (695, 377), FACE_BBOX)
        assert result['final_pitch'] > 3, f"pitch={result['final_pitch']:.1f}"

    def test_look_up_negative_pitch(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        ge.reset()
        result = ge.estimate(landmarks, (585, 363), (695, 363), FACE_BBOX)
        assert result['final_pitch'] < -3, f"pitch={result['final_pitch']:.1f}"

    def test_extreme_down_large_pitch(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        ge.reset()
        result = ge.estimate(landmarks, (585, 382), (695, 382), FACE_BBOX)
        assert result['final_pitch'] > 15, f"pitch={result['final_pitch']:.1f}"

    def test_extreme_down_off_road_zone(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        ge.reset()
        result = ge.estimate(landmarks, (585, 382), (695, 382), FACE_BBOX)
        assert not result['is_on_road'], f"zone={result['zone_name']} type={result['zone_type']}"

    def test_zone_type_returned(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        ge.reset()
        result = ge.estimate(landmarks, (585, 382), (695, 382), FACE_BBOX)
        assert result['zone_type'] in ('on_road', 'off_road')

    def test_gaze_confidence_returned(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        ge.reset()
        result = ge.estimate(landmarks, (585, 382), (695, 382), FACE_BBOX)
        assert 0.0 <= result['gaze_confidence'] <= 1.0, f"confidence={result['gaze_confidence']:.2f}"

    def test_stabilization_low_jitter(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        ge.reset()
        yaws = []
        for i in range(10):
            result = ge.estimate(landmarks, (585, 370), (695, 370), FACE_BBOX, dt=0.04)
            yaws.append(result['final_yaw'])
        yaw_std = (sum((y - sum(yaws) / len(yaws)) ** 2 for y in yaws) / len(yaws)) ** 0.5
        assert yaw_std < 5.0, f"std={yaw_std:.2f}"

    def test_calibration_offset_shifts_yaw(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        ge.yaw_offset = 10.0
        ge.pitch_offset = 5.0
        ge.reset()
        result = ge.estimate(landmarks, (585, 370), (695, 370), FACE_BBOX)
        assert result['raw_yaw'] != result['final_yaw'] or ge.yaw_offset == 0

    def test_iris_ratio_x_centered(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        left_eye = landmarks[5:11]
        ratio_x, ratio_y = ge.compute_iris_ratio(left_eye, (585, 370))
        assert abs(ratio_x) < 0.3, f"x={ratio_x:.2f}"

    def test_iris_ratio_y_centered(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        left_eye = landmarks[5:11]
        ratio_x, ratio_y = ge.compute_iris_ratio(left_eye, (585, 370))
        assert abs(ratio_y) < 0.3, f"y={ratio_y:.2f}"

    def test_iris_ratio_x_shifted_left(self):
        ge = _make_gaze_estimator()
        landmarks = _make_landmarks()
        left_eye = landmarks[5:11]
        ratio_x_shifted, _ = ge.compute_iris_ratio(left_eye, (572, 370))
        assert ratio_x_shifted < -0.2, f"x={ratio_x_shifted:.2f}"


class TestGazeCalibration:
    def test_9_calibration_points_defined(self):
        assert len(CALIBRATION_POINTS) == 9

    def test_labels_correct(self):
        assert CALIBRATION_POINTS[0][0] == 'center'

    def test_center_expected_00(self):
        assert CALIBRATION_POINTS[0][1] == 0.0 and CALIBRATION_POINTS[0][2] == 0.0

    def test_collecting_after_start(self):
        gc = GazeCalibration(duration_per_point=0.5, fps=25)
        gc.start()
        assert gc.collecting

    def test_current_point_is_center(self):
        gc = GazeCalibration(duration_per_point=0.5, fps=25)
        gc.start()
        assert gc.current_label == 'center'

    def test_center_not_done_after_11_frames(self):
        gc = GazeCalibration(duration_per_point=0.5, fps=25)
        gc.start()
        for i in range(11):
            done = gc.update(2.0, 1.0)
        assert not done

    def test_center_done_after_12_frames(self):
        gc = GazeCalibration(duration_per_point=0.5, fps=25)
        gc.start()
        for i in range(11):
            gc.update(2.0, 1.0)
        done = gc.update(2.0, 1.0)
        assert done

    def test_moved_to_next_point(self):
        gc = GazeCalibration(duration_per_point=0.5, fps=25)
        gc.start()
        for i in range(12):
            gc.update(2.0, 1.0)
        assert gc.current_point == 1

    def test_next_point_is_far_left(self):
        gc = GazeCalibration(duration_per_point=0.5, fps=25)
        gc.start()
        for i in range(12):
            gc.update(2.0, 1.0)
        assert gc.current_label == 'far_left'

    def test_all_9_points_collected(self):
        gc = GazeCalibration(duration_per_point=0.5, fps=25)
        gc.start()
        for pt_idx in range(1, 9):
            frames_needed = int(gc.duration_per_point * gc.fps)
            for i in range(frames_needed):
                gc.update(0.0, 0.0)
        assert gc.current_point == 9

    def test_calibration_complete(self):
        gc = GazeCalibration(duration_per_point=0.5, fps=25)
        gc.start()
        for i in range(12):
            gc.update(2.0, 1.0)
        for pt_idx in range(1, 9):
            frames_needed = int(gc.duration_per_point * gc.fps)
            for i in range(frames_needed):
                gc.update(0.0, 0.0)
        assert gc.calibrated

    def test_yaw_offset_computed(self):
        gc = GazeCalibration(duration_per_point=0.5, fps=25)
        gc.start()
        for i in range(12):
            gc.update(2.0, 1.0)
        for pt_idx in range(1, 9):
            frames_needed = int(gc.duration_per_point * gc.fps)
            for i in range(frames_needed):
                gc.update(0.0, 0.0)
        assert gc.yaw_offset == 2.0, f"got {gc.yaw_offset}"

    def test_pitch_offset_computed(self):
        gc = GazeCalibration(duration_per_point=0.5, fps=25)
        gc.start()
        for i in range(12):
            gc.update(2.0, 1.0)
        for pt_idx in range(1, 9):
            frames_needed = int(gc.duration_per_point * gc.fps)
            for i in range(frames_needed):
                gc.update(0.0, 0.0)
        assert gc.pitch_offset == 1.0, f"got {gc.pitch_offset}"

    def test_calibration_save_load(self):
        gc = GazeCalibration(duration_per_point=0.5, fps=25)
        gc.start()
        for i in range(12):
            gc.update(2.0, 1.0)
        for pt_idx in range(1, 9):
            frames_needed = int(gc.duration_per_point * gc.fps)
            for i in range(frames_needed):
                gc.update(0.0, 0.0)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            tmppath = f.name
        gc.save(tmppath)
        gc2 = GazeCalibration()
        loaded = gc2.load(tmppath)
        assert loaded and gc2.yaw_offset == 2.0, f"loaded={loaded}"
        os.unlink(tmppath)


class TestBaselineLearner:
    def test_baseline_established_after_learning(self):
        bl = BaselineLearner(learning_period_sec=2.0, fps=25)
        t = time.monotonic()
        bl.start(timestamp=t)
        for i in range(100):
            t += 0.04
            bl.update(0.30, 0.30, is_blink=(i % 25 == 0), timestamp=t)
        assert bl.baseline_established

    def test_baseline_ear_open(self):
        bl = BaselineLearner(learning_period_sec=2.0, fps=25)
        t = time.monotonic()
        bl.start(timestamp=t)
        for i in range(100):
            t += 0.04
            bl.update(0.30, 0.30, is_blink=(i % 25 == 0), timestamp=t)
        assert 0.25 < bl.baseline_ear_open < 0.35, f"got {bl.baseline_ear_open:.3f}"

    def test_baseline_blink_rate_positive(self):
        bl = BaselineLearner(learning_period_sec=2.0, fps=25)
        t = time.monotonic()
        bl.start(timestamp=t)
        for i in range(100):
            t += 0.04
            bl.update(0.30, 0.30, is_blink=(i % 25 == 0), timestamp=t)
        assert bl.baseline_blink_rate > 0, f"got {bl.baseline_blink_rate:.1f}"

    def test_personalized_ear_threshold(self):
        bl = BaselineLearner(learning_period_sec=2.0, fps=25)
        t = time.monotonic()
        bl.start(timestamp=t)
        for i in range(100):
            t += 0.04
            bl.update(0.30, 0.30, is_blink=(i % 25 == 0), timestamp=t)
        assert 0.05 < bl.personalized_ear_close_threshold < 0.15, f"got {bl.personalized_ear_close_threshold:.3f}"

    def test_personalized_blink_rate_high(self):
        bl = BaselineLearner(learning_period_sec=2.0, fps=25)
        t = time.monotonic()
        bl.start(timestamp=t)
        for i in range(100):
            t += 0.04
            bl.update(0.30, 0.30, is_blink=(i % 25 == 0), timestamp=t)
        assert bl.personalized_blink_rate_high > bl.baseline_blink_rate, f"got {bl.personalized_blink_rate_high:.1f}"
