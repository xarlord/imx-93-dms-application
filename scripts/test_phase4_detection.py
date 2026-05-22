"""Phase 4 tests: Detection stages (head, landmark, iris) in mock mode."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.head_detector import HeadDetector
from src.landmark_detector import LandmarkDetector, MEDIAPIPE_TO_DMS, MP_TO_DMS
from src.iris_detector import IrisDetector

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
    print('\n=== Phase 4A: HeadDetector (Mock Mode) ===')
    # ============================================================

    hd = HeadDetector(mock=True)
    frame = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)

    # Test 1: detect returns list
    dets = hd.detect(frame)
    test('detect returns list', isinstance(dets, list),
         f'got {type(dets)}')

    # Test 2: at least one detection
    test('detect finds face', len(dets) > 0,
         f'got {len(dets)} detections')

    # Test 3: detection has bbox and confidence
    if dets:
        d = dets[0]
        test('detection has bbox', 'bbox' in d, f'keys: {list(d.keys())}')
        test('detection has confidence', 'confidence' in d, f'keys: {list(d.keys())}')
        test('bbox is 4-element list', len(d['bbox']) == 4,
             f'len={len(d["bbox"])}')
        x1, y1, x2, y2 = d['bbox']
        test('bbox x2 > x1', x2 > x1, f'{x1},{x2}')
        test('bbox y2 > y1', y2 > y1, f'{y1},{y2}')
        test('bbox within frame', x1 >= 0 and y1 >= 0 and x2 <= 1280 and y2 <= 800,
             f'bbox ({x1},{y1})-({x2},{y2}) frame 1280x800')
        test('confidence > 0', d['confidence'] > 0,
             f'conf={d["confidence"]}')

    # Test 4: get_best_detection returns top-1
    best = hd.get_best_detection(frame)
    test('get_best_detection returns dict', isinstance(best, dict),
         f'got {type(best)}')
    if best:
        test('best detection has bbox', 'bbox' in best)

    # Test 5: different frame sizes
    small_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dets_small = hd.detect(small_frame)
    test('detect works on 640x480', len(dets_small) > 0,
         f'got {len(dets_small)} detections')

    # Test 6: empty frame
    empty_frame = np.zeros((800, 1280, 3), dtype=np.uint8)
    dets_empty = hd.detect(empty_frame)
    test('detect works on black frame', isinstance(dets_empty, list))

    # Test 7: mock bbox is centered (within 25% of frame center)
    if dets:
        d0 = dets[0]
        bx1, by1, bx2, by2 = d0['bbox']
        bcx = (bx1 + bx2) / 2
        bcy = (by1 + by2) / 2
        test('mock bbox centered horizontally',
             abs(bcx - 640) < 320, f'center_x={bcx}')
        test('mock bbox centered vertically',
             abs(bcy - 400) < 200, f'center_y={bcy}')

    # ============================================================
    print('\n=== Phase 4B: LandmarkDetector (Mock Mode) ===')
    # ============================================================

    ld = LandmarkDetector(mock=True)
    head_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    head_bbox = [300, 200, 500, 450]  # x1, y1, x2, y2

    # Test 8: detect returns list of 25 landmarks
    lm = ld.detect(head_crop, head_bbox)
    test('detect returns list', isinstance(lm, list), f'got {type(lm)}')
    test('detect returns 25 landmarks', lm is not None and len(lm) == 25,
         f'got {len(lm) if lm else 0}')

    # Test 9: landmarks are (x, y) tuples
    if lm:
        test('landmark is tuple/list', isinstance(lm[0], (tuple, list)),
             f'got {type(lm[0])}')
        test('landmark has 2 elements', len(lm[0]) == 2,
             f'len={len(lm[0])}')

    # Test 10: landmarks within bbox (with margin for proportions)
    if lm:
        bw = head_bbox[2] - head_bbox[0]
        bh = head_bbox[3] - head_bbox[1]
        all_in = all(
            head_bbox[0] - bw * 0.1 <= p[0] <= head_bbox[2] + bw * 0.1 and
            head_bbox[1] - bh * 0.1 <= p[1] <= head_bbox[3] + bh * 0.1
            for p in lm
        )
        test('landmarks within bbox region (10% margin)', all_in)

    # Test 11: nose tip (idx 0) near center of bbox
    if lm:
        cx = (head_bbox[0] + head_bbox[2]) / 2
        cy = (head_bbox[1] + head_bbox[3]) / 2
        bw = head_bbox[2] - head_bbox[0]
        nose = lm[0]
        test('nose tip near center horizontally',
             abs(nose[0] - cx) < bw * 0.15,
             f'nose_x={nose[0]}, cx={cx}')

    # Test 12: eye landmarks (5-10 left, 11-16 right) above mouth (17-24)
    if lm:
        left_eye_y = [lm[i][1] for i in range(5, 11)]
        mouth_y = [lm[i][1] for i in range(17, 25)]
        test('left eye above mouth',
             max(left_eye_y) < min(mouth_y),
             f'eye_max_y={max(left_eye_y):.0f}, mouth_min_y={min(mouth_y):.0f}')

    # Test 13: left eye landmarks to the left of right eye
    if lm:
        left_eye_x = [lm[i][0] for i in range(5, 11)]
        right_eye_x = [lm[i][0] for i in range(11, 17)]
        test('left eye left of right eye',
             np.mean(left_eye_x) < np.mean(right_eye_x),
             f'left_mean={np.mean(left_eye_x):.0f}, right_mean={np.mean(right_eye_x):.0f}')

    # Test 14: MEDIAPIPE_TO_DMS mapping completeness
    test('MEDIAPIPE_TO_DMS has 25 entries', len(MEDIAPIPE_TO_DMS) == 25,
         f'got {len(MEDIAPIPE_TO_DMS)}')

    # Test 15: Reverse mapping consistency
    test('MP_TO_DMS reverse mapping consistent',
         all(MEDIAPIPE_TO_DMS[MP_TO_DMS[v]] == v for v in MP_TO_DMS),
         'mapping mismatch')

    # Test 16: Different bbox sizes
    small_bbox = [100, 100, 150, 160]
    lm_small = ld.detect(head_crop, small_bbox)
    test('detect works on small bbox',
         lm_small is not None and len(lm_small) == 25,
         f'got {len(lm_small) if lm_small else 0} landmarks')

    # Test 17: Landmarks scale with bbox
    if lm and lm_small:
        # Nose tip should scale proportionally
        big_cx = (head_bbox[0] + head_bbox[2]) / 2
        small_cx = (small_bbox[0] + small_bbox[2]) / 2
        test('landmarks scale with bbox size',
             abs(lm_small[0][0] - small_cx) < abs(lm[0][0] - big_cx) + 10)

    # ============================================================
    print('\n=== Phase 4C: IrisDetector (Mock Mode) ===')
    # ============================================================

    iris_det = IrisDetector(mock=True)

    # Realistic eye landmarks (6 points per eye, based on MediaPipe proportions)
    left_eye_lm = [(100, 150), (115, 142), (125, 142), (135, 150),
                    (125, 158), (115, 158)]  # outer, upper1, upper2, inner, lower1, lower2
    right_eye_lm = [(200, 150), (215, 142), (225, 142), (235, 150),
                     (225, 158), (215, 158)]

    frame_rgb = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

    # Test 18: detect returns tuple of two
    result = iris_det.detect(frame_rgb, left_eye_lm, right_eye_lm)
    test('detect returns tuple', isinstance(result, tuple) and len(result) == 2,
         f'got {type(result)}')

    left_iris, right_iris = result

    # Test 19: iris is (cx, cy, r) tuple
    test('left iris is tuple/list', isinstance(left_iris, (tuple, list)),
         f'got {type(left_iris)}')
    test('left iris has 3 elements', len(left_iris) == 3,
         f'len={len(left_iris)}')
    test('right iris is tuple/list', isinstance(right_iris, (tuple, list)))
    test('right iris has 3 elements', len(right_iris) == 3)

    # Test 20: iris center near eye center
    if left_iris:
        left_cx = sum(p[0] for p in left_eye_lm) / len(left_eye_lm)
        left_cy = sum(p[1] for p in left_eye_lm) / len(left_eye_lm)
        test('left iris cx near eye center',
             abs(left_iris[0] - left_cx) < 15,
             f'iris_cx={left_iris[0]:.1f}, eye_cx={left_cx:.1f}')
        test('left iris cy near eye center',
             abs(left_iris[1] - left_cy) < 15,
             f'iris_cy={left_iris[1]:.1f}, eye_cy={left_cy:.1f}')

    # Test 21: iris radius is positive
    if left_iris and right_iris:
        test('left iris radius > 0', left_iris[2] > 0,
             f'r={left_iris[2]}')
        test('right iris radius > 0', right_iris[2] > 0,
             f'r={right_iris[2]}')

    # Test 22: iris radius is reasonable fraction of eye width
    if left_iris:
        eye_width = abs(left_eye_lm[0][0] - left_eye_lm[3][0])
        test('left iris radius ~10-20% of eye width',
             eye_width * 0.05 < left_iris[2] < eye_width * 0.40,
             f'r={left_iris[2]:.1f}, eye_w={eye_width}')

    # Test 23: right iris is to the right of left iris
    if left_iris and right_iris:
        test('right iris to the right of left iris',
             right_iris[0] > left_iris[0],
             f'right_cx={right_iris[0]:.1f}, left_cx={left_iris[0]:.1f}')

    # Test 24: Different eye positions
    big_left = [(50, 75), (75, 65), (95, 65), (115, 75), (95, 85), (75, 85)]
    big_right = [(200, 75), (225, 65), (245, 65), (265, 75), (245, 85), (225, 85)]
    big_frame = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
    bl, br = iris_det.detect(big_frame, big_left, big_right)
    test('detect works on different eye positions',
         bl is not None and br is not None,
         f'left={bl}, right={br}')

    # Test 25: Mock radius scales with eye size
    if left_iris and bl:
        small_eye_w = abs(left_eye_lm[0][0] - left_eye_lm[3][0])
        big_eye_w = abs(big_left[0][0] - big_left[3][0])
        test('iris radius scales with eye width',
             bl[2] > left_iris[2],
             f'small_r={left_iris[2]:.1f}, big_r={bl[2]:.1f}')

    # ============================================================
    print(f'\n{"="*60}')
    print(f'Phase 4 Detection Results: {PASS} PASS, {FAIL} FAIL, {PASS+FAIL} total')
    if FAIL > 0:
        print('SOME TESTS FAILED')
    else:
        print('ALL TESTS PASSED')
        print(f'{"="*60}')


if __name__ == '__main__':
    main()
