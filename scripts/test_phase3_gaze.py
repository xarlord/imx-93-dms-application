"""Test Phase 3 gaze estimation, calibration, and baseline against spec."""
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

    # ── Gaze Estimator ──
    print("\n=== gaze_estimator.py ===")

    zones_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'zones.xml')
    zl = ZoneLoader(zones_path)
    ge = GazeEstimator(yaw_weight=32.0, pitch_weight=45.0, ppd_yaw=5.0, ppd_pitch=5.0,
                       display_w=1920, display_h=1080)
    ge.set_zone_loader(zl)

    # Synthetic face: centered in 1280x800 frame, landmarks at known positions
    # Face bbox centered at (640, 400), width 200, height 250
    face_bbox = [540, 275, 740, 525]
    # 25 landmarks: nose at center, eyes above, mouth below
    # Left eye centered at (590, 370), Right eye centered at (690, 370)
    landmarks = [(640, 400)]  # 0: nose tip (centered)
    landmarks += [(580, 360), (600, 360)]  # 1-2: left eyebrow
    landmarks += [(680, 360), (700, 360)]  # 3-4: right eyebrow
    # Left eye: outer(570,370) upper1(580,362) upper2(595,362) inner(600,370) lower1(595,378) lower2(580,378)
    landmarks += [(570, 370), (580, 362), (595, 362), (600, 370), (595, 378), (580, 378)]
    # Right eye: outer(710,370) upper1(700,362) upper2(685,362) inner(680,370) lower1(685,378) lower2(700,378)
    landmarks += [(710, 370), (700, 362), (685, 362), (680, 370), (685, 378), (700, 378)]
    # Mouth: 8 points forming a closed mouth
    landmarks += [(610, 430), (625, 425), (640, 424), (655, 425),
                  (670, 430), (655, 436), (640, 437), (625, 436)]

    assert len(landmarks) == 25, f"Expected 25 landmarks, got {len(landmarks)}"

    # Test 1: Looking straight ahead (iris at eye center)
    left_iris = (585, 370)   # centered in left eye
    right_iris = (695, 370)  # centered in right eye

    result = ge.estimate(landmarks, left_iris, right_iris, face_bbox)
    test("Straight ahead: small yaw", abs(result['final_yaw']) < 10, f"yaw={result['final_yaw']:.1f}")
    test("Straight ahead: small pitch", abs(result['final_pitch']) < 10, f"pitch={result['final_pitch']:.1f}")
    test("Straight ahead: ROAD_AHEAD zone", result['zone_name'] == 'ROAD_AHEAD',
         f"zone={result['zone_name']}")

    # Test 2: Looking left (both irises shifted screen-left in each eye)
    ge.reset()
    left_iris = (572, 370)   # shifted toward outer (screen-left) in left eye
    right_iris = (682, 370)  # shifted toward inner (screen-left) in right eye
    result = ge.estimate(landmarks, left_iris, right_iris, face_bbox)
    test("Look left: negative yaw", result['final_yaw'] < -3, f"yaw={result['final_yaw']:.1f}")

    # Test 3: Looking right (both irises shifted screen-right in each eye)
    ge.reset()
    left_iris = (598, 370)   # shifted toward inner (screen-right) in left eye
    right_iris = (708, 370)  # shifted toward outer (screen-right) in right eye
    result = ge.estimate(landmarks, left_iris, right_iris, face_bbox)
    test("Look right: positive yaw", result['final_yaw'] > 3, f"yaw={result['final_yaw']:.1f}")

    # Test 4: Looking down (iris shifted down in eye)
    ge.reset()
    left_iris = (585, 377)   # shifted down
    right_iris = (695, 377)
    result = ge.estimate(landmarks, left_iris, right_iris, face_bbox)
    test("Look down: positive pitch", result['final_pitch'] > 3, f"pitch={result['final_pitch']:.1f}")

    # Test 5: Looking up (iris shifted up)
    ge.reset()
    left_iris = (585, 363)
    right_iris = (695, 363)
    result = ge.estimate(landmarks, left_iris, right_iris, face_bbox)
    test("Look up: negative pitch", result['final_pitch'] < -3, f"pitch={result['final_pitch']:.1f}")

    # Test 6: Extreme gaze down -> should hit LOWER_DASH zone
    ge.reset()
    left_iris = (585, 382)
    right_iris = (695, 382)
    result = ge.estimate(landmarks, left_iris, right_iris, face_bbox)
    test("Extreme down: large pitch", result['final_pitch'] > 15, f"pitch={result['final_pitch']:.1f}")

    # Test 7: Zone classification — extreme down should be off-road
    test("Extreme down: off-road zone", not result['is_on_road'],
         f"zone={result['zone_name']} type={result['zone_type']}")
    test("Zone type returned", result['zone_type'] in ('on_road', 'off_road'))
    test("Gaze confidence returned", 0.0 <= result['gaze_confidence'] <= 1.0,
         f"confidence={result['gaze_confidence']:.2f}")

    # Test 8: Stabilization - successive similar frames converge
    ge.reset()
    yaws = []
    for i in range(10):
        left_iris = (585, 370)
        right_iris = (695, 370)
        result = ge.estimate(landmarks, left_iris, right_iris, face_bbox, dt=0.04)
        yaws.append(result['final_yaw'])
    yaw_std = (sum((y - sum(yaws)/len(yaws))**2 for y in yaws) / len(yaws)) ** 0.5
    test("Stabilization: low jitter (std < 5 deg)", yaw_std < 5.0, f"std={yaw_std:.2f}")

    # Test 9: Calibration offsets applied
    ge.yaw_offset = 10.0
    ge.pitch_offset = 5.0
    ge.reset()
    result = ge.estimate(landmarks, (585, 370), (695, 370), face_bbox)
    test("Calibration offset shifts yaw", result['raw_yaw'] != result['final_yaw'] or ge.yaw_offset == 0)
    ge.yaw_offset = 0.0
    ge.pitch_offset = 0.0

    # Test 10: Iris ratio function
    left_eye = landmarks[5:11]
    ratio_x, ratio_y = ge.compute_iris_ratio(left_eye, (585, 370))
    test("Iris ratio X centered", abs(ratio_x) < 0.3, f"x={ratio_x:.2f}")
    test("Iris ratio Y centered", abs(ratio_y) < 0.3, f"y={ratio_y:.2f}")

    ratio_x_shifted, _ = ge.compute_iris_ratio(left_eye, (572, 370))
    test("Iris ratio X shifted left", ratio_x_shifted < -0.2, f"x={ratio_x_shifted:.2f}")

    # ── Gaze Calibration ──
    print("\n=== gaze_calibration.py ===")

    gc = GazeCalibration(duration_per_point=0.5, fps=25)

    test("9 calibration points defined", len(CALIBRATION_POINTS) == 9)
    test("Labels correct", CALIBRATION_POINTS[0][0] == 'center')
    test("Center expected (0,0)",
         CALIBRATION_POINTS[0][1] == 0.0 and CALIBRATION_POINTS[0][2] == 0.0)

    gc.start()
    test("Collecting after start", gc.collecting)
    test("Current point is center", gc.current_label == 'center')

    # Feed samples for center point (0.5s * 25fps = 12 frames needed)
    for i in range(11):
        done = gc.update(2.0, 1.0)  # simulate raw gaze with offset
    test("Center point not done after 11 frames", not done)

    done = gc.update(2.0, 1.0)  # 12th frame completes
    test("Center point done after 12 frames", done)
    test("Moved to next point", gc.current_point == 1)
    test("Next point is far_left", gc.current_label == 'far_left')

    # Complete remaining points quickly
    for pt_idx in range(1, 9):
        frames_needed = int(gc.duration_per_point * gc.fps)
        for i in range(frames_needed):
            gc.update(0.0, 0.0)

    test("All 9 points collected", gc.current_point == 9)
    test("Calibration complete", gc.calibrated)
    test("Yaw offset computed", gc.yaw_offset == 2.0, f"got {gc.yaw_offset}")
    test("Pitch offset computed", gc.pitch_offset == 1.0, f"got {gc.pitch_offset}")

    # Save/load
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        tmppath = f.name
    gc.save(tmppath)
    gc2 = GazeCalibration()
    loaded = gc2.load(tmppath)
    test("Calibration save/load", loaded and gc2.yaw_offset == 2.0, f"loaded={loaded}")
    os.unlink(tmppath)

    # ── Baseline Learner ──
    print("\n=== baseline.py ===")

    bl = BaselineLearner(learning_period_sec=2.0, fps=25)  # Short for testing
    t = time.monotonic()
    bl.start(timestamp=t)

    # Feed normal driving data with timestamps simulating 4 seconds elapsed
    for i in range(100):  # 4 seconds at 25fps
        t += 0.04
        bl.update(0.30, 0.30, is_blink=(i % 25 == 0), timestamp=t)

    test("Baseline established after learning period", bl.baseline_established)
    test("Baseline EAR open ~0.30", 0.25 < bl.baseline_ear_open < 0.35,
         f"got {bl.baseline_ear_open:.3f}")
    test("Baseline blink rate > 0", bl.baseline_blink_rate > 0,
         f"got {bl.baseline_blink_rate:.1f}")
    test("Personalized EAR threshold", 0.05 < bl.personalized_ear_close_threshold < 0.15,
         f"got {bl.personalized_ear_close_threshold:.3f}")
    test("Personalized blink rate high", bl.personalized_blink_rate_high > bl.baseline_blink_rate,
         f"got {bl.personalized_blink_rate_high:.1f}")

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"Phase 3 Results: {PASS} PASS, {FAIL} FAIL out of {PASS+FAIL}")
    if FAIL > 0:
        print("FAILURES DETECTED - fix before proceeding!")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == '__main__':
    main()
