import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.head_detector import HeadDetector
from src.landmark_detector import LandmarkDetector, MEDIAPIPE_TO_DMS, MP_TO_DMS
from src.iris_detector import IrisDetector


class TestHeadDetector:
    def test_detect_returns_list(self):
        hd = HeadDetector(mock=True)
        frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        dets = hd.detect(frame)
        assert isinstance(dets, list), f'got {type(dets)}'

    def test_detect_finds_face(self):
        hd = HeadDetector(mock=True)
        frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        dets = hd.detect(frame)
        assert len(dets) > 0, f'got {len(dets)} detections'

    def test_detection_has_bbox(self):
        hd = HeadDetector(mock=True)
        frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        dets = hd.detect(frame)
        if dets:
            assert 'bbox' in dets[0], f'keys: {list(dets[0].keys())}'

    def test_detection_has_confidence(self):
        hd = HeadDetector(mock=True)
        frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        dets = hd.detect(frame)
        if dets:
            assert 'confidence' in dets[0], f'keys: {list(dets[0].keys())}'

    def test_bbox_is_4_element_list(self):
        hd = HeadDetector(mock=True)
        frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        dets = hd.detect(frame)
        if dets:
            assert len(dets[0]['bbox']) == 4, f'len={len(dets[0]["bbox"])}'

    def test_bbox_x2_greater_x1(self):
        hd = HeadDetector(mock=True)
        frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        dets = hd.detect(frame)
        if dets:
            x1, y1, x2, y2 = dets[0]['bbox']
            assert x2 > x1, f'{x1},{x2}'

    def test_bbox_y2_greater_y1(self):
        hd = HeadDetector(mock=True)
        frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        dets = hd.detect(frame)
        if dets:
            x1, y1, x2, y2 = dets[0]['bbox']
            assert y2 > y1, f'{y1},{y2}'

    def test_bbox_within_frame(self):
        hd = HeadDetector(mock=True)
        frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        dets = hd.detect(frame)
        if dets:
            x1, y1, x2, y2 = dets[0]['bbox']
            assert x1 >= 0 and y1 >= 0 and x2 <= 1280 and y2 <= 800, f'bbox ({x1},{y1})-({x2},{y2}) frame 1280x800'

    def test_confidence_positive(self):
        hd = HeadDetector(mock=True)
        frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        dets = hd.detect(frame)
        if dets:
            assert dets[0]['confidence'] > 0, f'conf={dets[0]["confidence"]}'

    def test_get_best_detection_returns_dict(self):
        hd = HeadDetector(mock=True)
        frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        best = hd.get_best_detection(frame)
        assert isinstance(best, dict), f'got {type(best)}'

    def test_best_detection_has_bbox(self):
        hd = HeadDetector(mock=True)
        frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        best = hd.get_best_detection(frame)
        if best:
            assert 'bbox' in best

    def test_detect_works_on_640x480(self):
        hd = HeadDetector(mock=True)
        small_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets_small = hd.detect(small_frame)
        assert len(dets_small) > 0, f'got {len(dets_small)} detections'

    def test_detect_works_on_black_frame(self):
        hd = HeadDetector(mock=True)
        empty_frame = np.zeros((800, 1280, 3), dtype=np.uint8)
        dets_empty = hd.detect(empty_frame)
        assert isinstance(dets_empty, list)

    def test_mock_bbox_centered_horizontally(self):
        hd = HeadDetector(mock=True)
        frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        dets = hd.detect(frame)
        if dets:
            d0 = dets[0]
            bx1, by1, bx2, by2 = d0['bbox']
            bcx = (bx1 + bx2) / 2
            assert abs(bcx - 640) < 320, f'center_x={bcx}'

    def test_mock_bbox_centered_vertically(self):
        hd = HeadDetector(mock=True)
        frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        dets = hd.detect(frame)
        if dets:
            d0 = dets[0]
            bx1, by1, bx2, by2 = d0['bbox']
            bcy = (by1 + by2) / 2
            assert abs(bcy - 400) < 200, f'center_y={bcy}'


class TestLandmarkDetector:
    def test_detect_returns_list(self):
        ld = LandmarkDetector(mock=True)
        head_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        head_bbox = [300, 200, 500, 450]
        lm = ld.detect(head_crop, head_bbox)
        assert isinstance(lm, list), f'got {type(lm)}'

    def test_detect_returns_25_landmarks(self):
        ld = LandmarkDetector(mock=True)
        head_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        head_bbox = [300, 200, 500, 450]
        lm = ld.detect(head_crop, head_bbox)
        assert lm is not None and len(lm) == 25, f'got {len(lm) if lm else 0}'

    def test_landmark_is_tuple_or_list(self):
        ld = LandmarkDetector(mock=True)
        head_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        head_bbox = [300, 200, 500, 450]
        lm = ld.detect(head_crop, head_bbox)
        if lm:
            assert isinstance(lm[0], (tuple, list)), f'got {type(lm[0])}'

    def test_landmark_has_2_elements(self):
        ld = LandmarkDetector(mock=True)
        head_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        head_bbox = [300, 200, 500, 450]
        lm = ld.detect(head_crop, head_bbox)
        if lm:
            assert len(lm[0]) == 2, f'len={len(lm[0])}'

    def test_landmarks_within_bbox_region(self):
        ld = LandmarkDetector(mock=True)
        head_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        head_bbox = [300, 200, 500, 450]
        lm = ld.detect(head_crop, head_bbox)
        if lm:
            bw = head_bbox[2] - head_bbox[0]
            bh = head_bbox[3] - head_bbox[1]
            all_in = all(
                head_bbox[0] - bw * 0.1 <= p[0] <= head_bbox[2] + bw * 0.1 and
                head_bbox[1] - bh * 0.1 <= p[1] <= head_bbox[3] + bh * 0.1
                for p in lm
            )
            assert all_in

    def test_nose_tip_near_center_horizontally(self):
        ld = LandmarkDetector(mock=True)
        head_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        head_bbox = [300, 200, 500, 450]
        lm = ld.detect(head_crop, head_bbox)
        if lm:
            cx = (head_bbox[0] + head_bbox[2]) / 2
            bw = head_bbox[2] - head_bbox[0]
            nose = lm[0]
            assert abs(nose[0] - cx) < bw * 0.15, f'nose_x={nose[0]}, cx={cx}'

    def test_left_eye_above_mouth(self):
        ld = LandmarkDetector(mock=True)
        head_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        head_bbox = [300, 200, 500, 450]
        lm = ld.detect(head_crop, head_bbox)
        if lm:
            left_eye_y = [lm[i][1] for i in range(5, 11)]
            mouth_y = [lm[i][1] for i in range(17, 25)]
            assert max(left_eye_y) < min(mouth_y), f'eye_max_y={max(left_eye_y):.0f}, mouth_min_y={min(mouth_y):.0f}'

    def test_left_eye_left_of_right_eye(self):
        ld = LandmarkDetector(mock=True)
        head_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        head_bbox = [300, 200, 500, 450]
        lm = ld.detect(head_crop, head_bbox)
        if lm:
            left_eye_x = [lm[i][0] for i in range(5, 11)]
            right_eye_x = [lm[i][0] for i in range(11, 17)]
            assert np.mean(left_eye_x) < np.mean(right_eye_x), f'left_mean={np.mean(left_eye_x):.0f}, right_mean={np.mean(right_eye_x):.0f}'

    def test_mediapipe_to_dms_has_25_entries(self):
        assert len(MEDIAPIPE_TO_DMS) == 25, f'got {len(MEDIAPIPE_TO_DMS)}'

    def test_mp_to_dms_reverse_mapping_consistent(self):
        assert all(MEDIAPIPE_TO_DMS[MP_TO_DMS[v]] == v for v in MP_TO_DMS), 'mapping mismatch'

    def test_detect_works_on_small_bbox(self):
        ld = LandmarkDetector(mock=True)
        head_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        small_bbox = [100, 100, 150, 160]
        lm_small = ld.detect(head_crop, small_bbox)
        assert lm_small is not None and len(lm_small) == 25, f'got {len(lm_small) if lm_small else 0} landmarks'

    def test_landmarks_scale_with_bbox_size(self):
        ld = LandmarkDetector(mock=True)
        head_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        head_bbox = [300, 200, 500, 450]
        small_bbox = [100, 100, 150, 160]
        lm = ld.detect(head_crop, head_bbox)
        lm_small = ld.detect(head_crop, small_bbox)
        if lm and lm_small:
            big_cx = (head_bbox[0] + head_bbox[2]) / 2
            small_cx = (small_bbox[0] + small_bbox[2]) / 2
            assert abs(lm_small[0][0] - small_cx) < abs(lm[0][0] - big_cx) + 10


class TestIrisDetector:
    @pytest.fixture()
    def iris_setup(self):
        iris_det = IrisDetector(mock=True)
        left_eye_lm = [(100, 150), (115, 142), (125, 142), (135, 150),
                        (125, 158), (115, 158)]
        right_eye_lm = [(200, 150), (215, 142), (225, 142), (235, 150),
                         (225, 158), (215, 158)]
        frame_rgb = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        return iris_det, left_eye_lm, right_eye_lm, frame_rgb

    def test_detect_returns_tuple_of_two(self, iris_setup):
        iris_det, left_eye_lm, right_eye_lm, frame_rgb = iris_setup
        result = iris_det.detect(frame_rgb, left_eye_lm, right_eye_lm)
        assert isinstance(result, tuple) and len(result) == 2, f'got {type(result)}'

    def test_left_iris_is_tuple_or_list(self, iris_setup):
        iris_det, left_eye_lm, right_eye_lm, frame_rgb = iris_setup
        left_iris, right_iris = iris_det.detect(frame_rgb, left_eye_lm, right_eye_lm)
        assert isinstance(left_iris, (tuple, list)), f'got {type(left_iris)}'

    def test_left_iris_has_3_elements(self, iris_setup):
        iris_det, left_eye_lm, right_eye_lm, frame_rgb = iris_setup
        left_iris, right_iris = iris_det.detect(frame_rgb, left_eye_lm, right_eye_lm)
        assert len(left_iris) == 3, f'len={len(left_iris)}'

    def test_right_iris_is_tuple_or_list(self, iris_setup):
        iris_det, left_eye_lm, right_eye_lm, frame_rgb = iris_setup
        left_iris, right_iris = iris_det.detect(frame_rgb, left_eye_lm, right_eye_lm)
        assert isinstance(right_iris, (tuple, list))

    def test_right_iris_has_3_elements(self, iris_setup):
        iris_det, left_eye_lm, right_eye_lm, frame_rgb = iris_setup
        left_iris, right_iris = iris_det.detect(frame_rgb, left_eye_lm, right_eye_lm)
        assert len(right_iris) == 3

    def test_left_iris_cx_near_eye_center(self, iris_setup):
        iris_det, left_eye_lm, right_eye_lm, frame_rgb = iris_setup
        left_iris, right_iris = iris_det.detect(frame_rgb, left_eye_lm, right_eye_lm)
        if left_iris:
            left_cx = sum(p[0] for p in left_eye_lm) / len(left_eye_lm)
            assert abs(left_iris[0] - left_cx) < 15, f'iris_cx={left_iris[0]:.1f}, eye_cx={left_cx:.1f}'

    def test_left_iris_cy_near_eye_center(self, iris_setup):
        iris_det, left_eye_lm, right_eye_lm, frame_rgb = iris_setup
        left_iris, right_iris = iris_det.detect(frame_rgb, left_eye_lm, right_eye_lm)
        if left_iris:
            left_cy = sum(p[1] for p in left_eye_lm) / len(left_eye_lm)
            assert abs(left_iris[1] - left_cy) < 15, f'iris_cy={left_iris[1]:.1f}, eye_cy={left_cy:.1f}'

    def test_left_iris_radius_positive(self, iris_setup):
        iris_det, left_eye_lm, right_eye_lm, frame_rgb = iris_setup
        left_iris, right_iris = iris_det.detect(frame_rgb, left_eye_lm, right_eye_lm)
        if left_iris and right_iris:
            assert left_iris[2] > 0, f'r={left_iris[2]}'

    def test_right_iris_radius_positive(self, iris_setup):
        iris_det, left_eye_lm, right_eye_lm, frame_rgb = iris_setup
        left_iris, right_iris = iris_det.detect(frame_rgb, left_eye_lm, right_eye_lm)
        if left_iris and right_iris:
            assert right_iris[2] > 0, f'r={right_iris[2]}'

    def test_left_iris_radius_reasonable_fraction(self, iris_setup):
        iris_det, left_eye_lm, right_eye_lm, frame_rgb = iris_setup
        left_iris, right_iris = iris_det.detect(frame_rgb, left_eye_lm, right_eye_lm)
        if left_iris:
            eye_width = abs(left_eye_lm[0][0] - left_eye_lm[3][0])
            assert eye_width * 0.05 < left_iris[2] < eye_width * 0.40, f'r={left_iris[2]:.1f}, eye_w={eye_width}'

    def test_right_iris_to_right_of_left(self, iris_setup):
        iris_det, left_eye_lm, right_eye_lm, frame_rgb = iris_setup
        left_iris, right_iris = iris_det.detect(frame_rgb, left_eye_lm, right_eye_lm)
        if left_iris and right_iris:
            assert right_iris[0] > left_iris[0], f'right_cx={right_iris[0]:.1f}, left_cx={left_iris[0]:.1f}'

    def test_detect_works_on_different_eye_positions(self, iris_setup):
        iris_det, _, _, _ = iris_setup
        big_left = [(50, 75), (75, 65), (95, 65), (115, 75), (95, 85), (75, 85)]
        big_right = [(200, 75), (225, 65), (245, 65), (265, 75), (245, 85), (225, 85)]
        big_frame = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
        bl, br = iris_det.detect(big_frame, big_left, big_right)
        assert bl is not None and br is not None, f'left={bl}, right={br}'

    def test_iris_radius_scales_with_eye_width(self, iris_setup):
        iris_det, left_eye_lm, right_eye_lm, frame_rgb = iris_setup
        left_iris, right_iris = iris_det.detect(frame_rgb, left_eye_lm, right_eye_lm)
        big_left = [(50, 75), (75, 65), (95, 65), (115, 75), (95, 85), (75, 85)]
        big_right = [(200, 75), (225, 65), (245, 65), (265, 75), (245, 85), (225, 85)]
        big_frame = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
        bl, br = iris_det.detect(big_frame, big_left, big_right)
        if left_iris and bl:
            small_eye_w = abs(left_eye_lm[0][0] - left_eye_lm[3][0])
            big_eye_w = abs(big_left[0][0] - big_left[3][0])
            assert bl[2] > left_iris[2], f'small_r={left_iris[2]:.1f}, big_r={bl[2]:.1f}'
