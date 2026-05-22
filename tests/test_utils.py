import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from src.utils.config import DMSConfig
from src.utils.geometry import (
    euclidean_distance, midpoint, compute_ear, compute_mar,
    point_in_polygon, classify_zone, polygon_area, compute_head_pose
)
from src.utils.image import (
    crop_with_padding, resize_for_model, crop_eye_region,
    scale_point_from_crop
)
from src.utils.zone_loader import ZoneLoader


class TestConfig:

    def test_camera_device(self, dms_config):
        assert dms_config.get('camera.device') == '/dev/video0'

    def test_camera_width(self, dms_config):
        assert dms_config.get('camera.width') == 1280

    def test_camera_height(self, dms_config):
        assert dms_config.get('camera.height') == 800

    def test_simulated_speed_fail_closed(self, dms_config):
        assert dms_config.get('vehicle.simulated_speed') == 0

    def test_ddaw_speed(self, dms_config):
        assert dms_config.get('vehicle.ddaw_activation_speed') == 70

    def test_addw_speed(self, dms_config):
        assert dms_config.get('vehicle.addw_activation_speed') == 20

    def test_yaw_weight(self, dms_config):
        assert dms_config.get('gaze.yaw_weight') == 32.0

    def test_pitch_weight(self, dms_config):
        assert dms_config.get('gaze.pitch_weight') == 45.0

    def test_ear_close_threshold(self, dms_config):
        assert dms_config.get('thresholds.ear_close') == 0.10

    def test_ear_blink_threshold(self, dms_config):
        assert dms_config.get('thresholds.ear_blink') == 0.19

    def test_yawn_mar_threshold(self, dms_config):
        assert dms_config.get('thresholds.yawn_mar') == 0.5

    def test_perclos_severe(self, dms_config):
        assert dms_config.get('thresholds.perclos_severe') == 0.20

    def test_perclos_mild(self, dms_config):
        assert dms_config.get('thresholds.perclos_mild') == 0.10

    def test_addw_high_speed_nominal(self, dms_config):
        assert dms_config.get('timing.addw_high_speed_nominal') == 3.5

    def test_addw_low_speed_nominal(self, dms_config):
        assert dms_config.get('timing.addw_low_speed_nominal') == 6.0

    def test_saccade_tolerance(self, dms_config):
        assert dms_config.get('timing.saccade_tolerance_ms') == 50

    def test_vats_window(self, dms_config):
        assert dms_config.get('timing.vats_window_sec') == 30

    def test_vats_threshold(self, dms_config):
        assert dms_config.get('timing.vats_threshold_sec') == 10.0

    def test_display_1920x1080(self, dms_config):
        assert dms_config.get('display.width') == 1920
        assert dms_config.get('display.height') == 1080

    def test_calibration_points(self, dms_config):
        assert dms_config.get('calibration.points') == 9

    def test_learning_period(self, dms_config):
        assert dms_config.get('calibration.learning_period_sec') == 600

    def test_config_section_accessor(self, dms_config):
        assert dms_config.camera['device'] == '/dev/video0'


class TestGeometry:

    def test_distance(self):
        assert abs(euclidean_distance((0, 0), (3, 4)) - 5.0) < 1e-6

    def test_midpoint(self):
        assert midpoint((0, 0), (4, 6)) == (2.0, 3.0)

    def test_ear_open_eye(self):
        open_eye = [(0, 3), (3, 1.5), (7, 1.5), (10, 3), (7, 4.5), (3, 4.5)]
        ear_open = compute_ear(open_eye)
        assert 0.25 < ear_open < 0.40, f"got {ear_open:.3f}"

    def test_ear_closed_eye(self):
        closed_eye = [(0, 3), (2, 2.8), (6, 2.8), (10, 3), (6, 3.2), (2, 3.2)]
        ear_closed = compute_ear(closed_eye)
        assert ear_closed < 0.10, f"got {ear_closed:.3f}"

    def test_mar_closed_mouth(self):
        closed_mouth = [(0, 5), (4, 4.8), (8, 4.8), (12, 4.8), (16, 5), (12, 5.2), (8, 5.2), (4, 5.2)]
        mar_closed = compute_mar(closed_mouth)
        assert mar_closed < 0.2, f"got {mar_closed:.3f}"

    def test_mar_yawn(self):
        yawn_mouth = [(0, 5), (3, 2), (6, 1), (10, 0), (10, 10), (6, 9), (3, 8), (0, 8)]
        mar_yawn = compute_mar(yawn_mouth)
        assert mar_yawn > 0.5, f"got {mar_yawn:.3f}"

    def test_point_in_polygon_inside(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((5, 5), square)

    def test_point_in_polygon_outside(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert not point_in_polygon((15, 5), square)

    def test_point_in_polygon_edge(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((0, 5), square)

    def test_head_yaw_reasonable(self):
        landmarks = [(70, 50)] + [(0, 0)] * 24
        face_bbox = [10, 10, 100, 100]
        yaw, pitch = compute_head_pose(landmarks, face_bbox)
        assert abs(yaw) < 90, f"got {yaw:.1f}"

    def test_head_pitch_reasonable(self):
        landmarks = [(70, 50)] + [(0, 0)] * 24
        face_bbox = [10, 10, 100, 100]
        yaw, pitch = compute_head_pose(landmarks, face_bbox)
        assert abs(pitch) < 90, f"got {pitch:.1f}"


class TestImage:

    def test_crop_shape(self, sample_frame):
        crop, bbox = crop_with_padding(sample_frame, [100, 100, 300, 300])
        assert crop.shape[0] > 200 and crop.shape[1] > 200, f"got {crop.shape}"

    def test_crop_bbox_clamped(self, sample_frame):
        crop, bbox = crop_with_padding(sample_frame, [100, 100, 300, 300])
        assert bbox[0] >= 0 and bbox[1] >= 0

    def test_resize_to_320x320(self, sample_frame):
        crop, bbox = crop_with_padding(sample_frame, [100, 100, 300, 300])
        resized, (scale, px, py) = resize_for_model(crop, (320, 320))
        assert resized.shape == (320, 320, 3), f"got {resized.shape}"

    def test_eye_crop_non_empty(self, sample_frame):
        eye_pts = [(100, 200), (110, 195), (120, 195), (130, 200), (120, 210), (110, 210)]
        eye_crop, eye_bbox = crop_eye_region(sample_frame, eye_pts)
        assert eye_crop.size > 0, f"got shape {eye_crop.shape}"

    def test_eye_resize_to_64x64(self, sample_frame):
        eye_pts = [(100, 200), (110, 195), (120, 195), (130, 200), (120, 210), (110, 210)]
        eye_crop, eye_bbox = crop_eye_region(sample_frame, eye_pts)
        eye_resized, _ = resize_for_model(eye_crop, (64, 64))
        assert eye_resized.shape == (64, 64, 3), f"got {eye_resized.shape}"


class TestZoneLoader:

    @pytest.fixture(autouse=True)
    def setup_zone_loader(self):
        zones_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'zones.xml')
        self.zl = ZoneLoader(zones_path)

    def test_loaded_16_zones(self):
        assert len(self.zl.zones) == 16, f"got {len(self.zl.zones)}"

    def test_center_road_ahead(self):
        z_id, z_name, z_type = self.zl.classify((960, 500))
        assert z_name == 'ROAD_AHEAD', f"got {z_name}"

    def test_bottom_center_lower_dash(self):
        z_id, z_name, z_type = self.zl.classify((960, 990))
        assert z_name == 'LOWER_DASH', f"got {z_name}"

    def test_left_mirror_area(self):
        z_id, z_name, z_type = self.zl.classify((150, 400))
        assert z_name == 'LEFT_MIRROR', f"got {z_name}"

    def test_road_ahead_is_on_road(self):
        assert self.zl.is_on_road(0)

    def test_lower_dash_is_off_road(self):
        assert not self.zl.is_on_road(6)

    def test_rear_mirror_is_on_road(self):
        assert self.zl.is_on_road(3)

    def test_top_center_rear_mirror_small_zone_first(self):
        z_id, z_name, z_type = self.zl.classify((960, 75))
        assert z_name == 'REAR_MIRROR', f"got {z_name}"
