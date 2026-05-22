"""Test Phase 1 utilities against architecture spec."""
import sys
import os
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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

PASS = 0
FAIL = 0


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {detail}")


def main():
    global PASS, FAIL

    # ── Config ──
    print("\n=== config.py ===")
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    cfg = DMSConfig(config_path)

    test("camera device", cfg.get('camera.device') == '/dev/video0')
    test("camera width", cfg.get('camera.width') == 1280)
    test("camera height", cfg.get('camera.height') == 800)
    test("simulated speed (fail-closed)", cfg.get('vehicle.simulated_speed') == 0)
    test("DDAW speed", cfg.get('vehicle.ddaw_activation_speed') == 70)
    test("ADDW speed", cfg.get('vehicle.addw_activation_speed') == 20)
    test("yaw weight", cfg.get('gaze.yaw_weight') == 32.0)
    test("pitch weight", cfg.get('gaze.pitch_weight') == 45.0)
    test("EAR close threshold", cfg.get('thresholds.ear_close') == 0.10)
    test("EAR blink threshold", cfg.get('thresholds.ear_blink') == 0.19)
    test("yawn MAR threshold", cfg.get('thresholds.yawn_mar') == 0.5)
    test("PERCLOS severe", cfg.get('thresholds.perclos_severe') == 0.20)
    test("PERCLOS mild", cfg.get('thresholds.perclos_mild') == 0.10)
    test("ADDW high speed nominal", cfg.get('timing.addw_high_speed_nominal') == 3.5)
    test("ADDW low speed nominal", cfg.get('timing.addw_low_speed_nominal') == 6.0)
    test("saccade tolerance", cfg.get('timing.saccade_tolerance_ms') == 50)
    test("VATS window", cfg.get('timing.vats_window_sec') == 30)
    test("VATS threshold", cfg.get('timing.vats_threshold_sec') == 10.0)
    test("display 1920x1080", cfg.get('display.width') == 1920 and cfg.get('display.height') == 1080)
    test("calibration points", cfg.get('calibration.points') == 9)
    test("learning period", cfg.get('calibration.learning_period_sec') == 600)
    test("config section accessor", cfg.camera['device'] == '/dev/video0')

    # ── Geometry ──
    print("\n=== geometry.py ===")
    test("distance", abs(euclidean_distance((0, 0), (3, 4)) - 5.0) < 1e-6)
    test("midpoint", midpoint((0, 0), (4, 6)) == (2.0, 3.0))

    # EAR: open eye should be ~0.3
    # 6 landmarks: [outer_corner, upper1, upper2, inner_corner, lower1, lower2]
    open_eye = [(0, 3), (3, 1.5), (7, 1.5), (10, 3), (7, 4.5), (3, 4.5)]
    ear_open = compute_ear(open_eye)
    test("EAR open eye ~0.3", 0.25 < ear_open < 0.40, f"got {ear_open:.3f}")

    # EAR: closed eye should be ~0.05
    closed_eye = [(0, 3), (2, 2.8), (6, 2.8), (10, 3), (6, 3.2), (2, 3.2)]
    ear_closed = compute_ear(closed_eye)
    test("EAR closed eye ~0.05", ear_closed < 0.10, f"got {ear_closed:.3f}")

    # MAR: closed mouth (upper and lower lips very close)
    # 8 landmarks: [left_corner, upper1, upper2, upper3, right_corner, lower1, lower2, lower3]
    closed_mouth = [(0, 5), (4, 4.8), (8, 4.8), (12, 4.8), (16, 5), (12, 5.2), (8, 5.2), (4, 5.2)]
    mar_closed = compute_mar(closed_mouth)
    test("MAR closed mouth", mar_closed < 0.2, f"got {mar_closed:.3f}")

    # MAR: open mouth (yawn)
    yawn_mouth = [(0, 5), (3, 2), (6, 1), (10, 0), (10, 10), (6, 9), (3, 8), (0, 8)]
    mar_yawn = compute_mar(yawn_mouth)
    test("MAR yawn >0.5", mar_yawn > 0.5, f"got {mar_yawn:.3f}")

    # Point in polygon
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    test("point in polygon (inside)", point_in_polygon((5, 5), square))
    test("point in polygon (outside)", not point_in_polygon((15, 5), square))
    test("point in polygon (edge)", point_in_polygon((0, 5), square))

    # Head pose
    landmarks = [(70, 50)] + [(0, 0)] * 24  # nose tip at (70, 50)
    face_bbox = [10, 10, 100, 100]
    yaw, pitch = compute_head_pose(landmarks, face_bbox)
    test("head yaw reasonable", abs(yaw) < 90, f"got {yaw:.1f}")
    test("head pitch reasonable", abs(pitch) < 90, f"got {pitch:.1f}")

    # ── Image ──
    print("\n=== image.py ===")
    import numpy as np

    frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)

    # crop_with_padding
    crop, bbox = crop_with_padding(frame, [100, 100, 300, 300])
    test("crop shape", crop.shape[0] > 200 and crop.shape[1] > 200, f"got {crop.shape}")
    test("crop bbox clamped", bbox[0] >= 0 and bbox[1] >= 0)

    # resize_for_model
    resized, (scale, px, py) = resize_for_model(crop, (320, 320))
    test("resize to 320x320", resized.shape == (320, 320, 3), f"got {resized.shape}")

    # crop_eye_region
    eye_pts = [(100, 200), (110, 195), (120, 195), (130, 200), (120, 210), (110, 210)]
    eye_crop, eye_bbox = crop_eye_region(frame, eye_pts)
    test("eye crop non-empty", eye_crop.size > 0, f"got shape {eye_crop.shape}")

    # resize_for_model small
    eye_resized, _ = resize_for_model(eye_crop, (64, 64))
    test("eye resize to 64x64", eye_resized.shape == (64, 64, 3), f"got {eye_resized.shape}")

    # ── Zone Loader ──
    print("\n=== zone_loader.py ===")
    zones_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'zones.xml')
    zl = ZoneLoader(zones_path)

    test("loaded 16 zones", len(zl.zones) == 16, f"got {len(zl.zones)}")

    # Road ahead zone (center of display)
    z_id, z_name, z_type = zl.classify((960, 500))
    test("center -> ROAD_AHEAD", z_name == 'ROAD_AHEAD', f"got {z_name}")

    # Lower dashboard
    z_id, z_name, z_type = zl.classify((960, 990))
    test("bottom center -> LOWER_DASH", z_name == 'LOWER_DASH', f"got {z_name}")

    # Left mirror
    z_id, z_name, z_type = zl.classify((150, 400))
    test("left mirror area", z_name == 'LEFT_MIRROR', f"got {z_name}")

    # is_on_road
    test("ROAD_AHEAD is on_road", zl.is_on_road(0))
    test("LOWER_DASH is off_road", not zl.is_on_road(6))
    test("REAR_MIRROR is on_road", zl.is_on_road(3))

    # Small zones match first (rear mirror is smaller than road ahead)
    z_id, z_name, z_type = zl.classify((960, 75))
    test("top center -> REAR_MIRROR (small zone first)", z_name == 'REAR_MIRROR', f"got {z_name}")

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"Phase 1 Results: {PASS} PASS, {FAIL} FAIL out of {PASS+FAIL}")
    if FAIL > 0:
        print("FAILURES DETECTED - fix before proceeding!")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == '__main__':
    main()
