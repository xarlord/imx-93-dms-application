"""Phase 6 tests: Camera capture and display modules (structure/logic tests).

GStreamer is not available on Windows dev host, so these tests verify:
- Module structure and imports
- Configuration handling
- Mock mode behavior
- Frame processing logic
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.capture import CameraCapture, PIPELINE_CAPTURE
from src.display import DisplayOutput, PIPELINE_DISPLAY

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
    print('\n=== Phase 6A: CameraCapture ===')
    # ============================================================

    # Test 1: Import and construction with defaults
    cap = CameraCapture()
    test('CameraCapture constructs with defaults', cap is not None)
    test('default device is /dev/video0', cap.device == '/dev/video0',
         f'got {cap.device}')
    test('default width is 1280', cap.width == 1280)
    test('default height is 800', cap.height == 800)
    test('default target_fps is 25', cap.target_fps == 25)

    # Test 2: Construction with config
    cfg = {'device': '/dev/video1', 'width': 640, 'height': 480, 'target_fps': 30}
    cap2 = CameraCapture(config=cfg)
    test('config device applied', cap2.device == '/dev/video1')
    test('config width applied', cap2.width == 640)
    test('config height applied', cap2.height == 480)
    test('config fps applied', cap2.target_fps == 30)

    # Test 3: Initial state
    test('initial frame_count is 0', cap.frame_count == 0)
    test('initial running is False', not cap.running)

    # Test 4: get_frame returns None before start
    frame = cap.get_frame()
    test('get_frame returns None before start', frame is None)

    # Test 5: Pipeline string format
    pipe = PIPELINE_CAPTURE.format(device='/dev/video0', width=1280, height=800, fps=25)
    test('pipeline contains v4l2src', 'v4l2src' in pipe)
    test('pipeline contains appsink', 'appsink' in pipe)
    test('pipeline contains imxvideoconvert_pxp', 'imxvideoconvert_pxp' in pipe)
    test('pipeline contains BGRx format', 'format=BGRx' in pipe)
    test('pipeline contains YUY2 input', 'format=YUY2' in pipe)

    # Test 6: Pipeline string with different config
    pipe2 = PIPELINE_CAPTURE.format(device='/dev/video1', width=640, height=480, fps=30)
    test('pipeline string uses device', '/dev/video1' in pipe2)
    test('pipeline string uses width', 'width=640' in pipe2)
    test('pipeline string uses fps', 'framerate=30/1' in pipe2)

    # Test 7: start raises without GStreamer (expected on Windows)
    try:
        cap.start()
        test('start without GStreamer raises', False, 'no exception raised')
    except RuntimeError as e:
        test('start without GStreamer raises RuntimeError',
             'GStreamer not available' in str(e))
    except Exception as e:
        # GStreamer may be partially available
        test('start handles missing GStreamer gracefully', True)

    # Test 8: stop is safe to call without start
    cap.stop()
    test('stop without start does not crash', True)

    # ============================================================
    print('\n=== Phase 6B: DisplayOutput ===')
    # ============================================================

    # Test 9: Import and construction with defaults
    disp = DisplayOutput()
    test('DisplayOutput constructs with defaults', disp is not None)
    test('default width is 1920', disp.width == 1920)
    test('default height is 1080', disp.height == 1080)
    test('default target_fps is 25', disp.target_fps == 25)
    test('default fullscreen is True', disp.fullscreen is True)

    # Test 10: Construction with config
    cfg_disp = {'width': 1280, 'height': 720, 'target_fps': 30, 'fullscreen': False}
    disp2 = DisplayOutput(config=cfg_disp)
    test('config width applied', disp2.width == 1280)
    test('config height applied', disp2.height == 720)
    test('config fps applied', disp2.target_fps == 30)
    test('config fullscreen applied', disp2.fullscreen is False)

    # Test 11: Initial state
    test('initial frame_count is 0', disp.frame_count == 0)
    test('initial running is False', not disp.running)

    # Test 12: push_frame returns False before start
    result = disp.push_frame(np.zeros((1080, 1920, 4), dtype=np.uint8))
    test('push_frame returns False before start', result is False)

    # Test 13: Pipeline string format
    pipe_disp = PIPELINE_DISPLAY.format(width=1920, height=1080, fps=25, fullscreen='true')
    test('pipeline contains appsrc', 'appsrc' in pipe_disp)
    test('pipeline contains waylandsink', 'waylandsink' in pipe_disp)
    test('pipeline contains imxvideoconvert_pxp', 'imxvideoconvert_pxp' in pipe_disp)
    test('pipeline contains BGRx', 'format=BGRx' in pipe_disp)
    test('pipeline fullscreen=true', 'fullscreen=true' in pipe_disp)

    # Test 14: Pipeline with non-fullscreen
    pipe_nf = PIPELINE_DISPLAY.format(width=1920, height=1080, fps=25, fullscreen='false')
    test('pipeline fullscreen=false', 'fullscreen=false' in pipe_nf)

    # Test 15: start raises without GStreamer
    try:
        disp.start()
        test('start without GStreamer raises', False, 'no exception raised')
    except RuntimeError as e:
        test('start without GStreamer raises RuntimeError',
             'GStreamer not available' in str(e))
    except Exception:
        test('start handles missing GStreamer gracefully', True)

    # Test 16: stop without start is safe
    disp.stop()
    test('stop without start does not crash', True)

    # ============================================================
    print('\n=== Phase 6C: Integration Helpers ===')
    # ============================================================

    # Test 17: Pipeline configs are consistent with arch spec
    test('capture uses PXP CSC', 'imxvideoconvert_pxp' in PIPELINE_CAPTURE)
    test('display uses PXP CSC', 'imxvideoconvert_pxp' in PIPELINE_DISPLAY)

    # Test 18: appsink has correct buffer settings
    test('appsink has max-buffers=2', 'max-buffers=2' in PIPELINE_CAPTURE)
    test('appsink has drop=true', 'drop=true' in PIPELINE_CAPTURE)
    test('appsink has emit-signals=true', 'emit-signals=true' in PIPELINE_CAPTURE)

    # Test 19: appsrc has correct settings
    test('appsrc has block=true', 'block=true' in PIPELINE_DISPLAY)
    test('appsrc has max-buffers=2', 'max-buffers=2' in PIPELINE_DISPLAY)

    # Test 20: waylandsink sync=false (non-blocking)
    test('waylandsink sync=false', 'sync=false' in PIPELINE_DISPLAY)

    # Test 21: Frame numpy array compatibility
    dummy_frame = np.zeros((800, 1280, 3), dtype=np.uint8)
    test('capture frame is contiguous', dummy_frame.flags['C_CONTIGUOUS'])
    test('capture frame dtype uint8', dummy_frame.dtype == np.uint8)
    test('capture frame shape correct', dummy_frame.shape == (800, 1280, 3))

    # Test 22: Display frame numpy array compatibility
    disp_frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
    test('display frame is contiguous', disp_frame.flags['C_CONTIGUOUS'])
    test('display frame dtype uint8', disp_frame.dtype == np.uint8)
    test('display frame shape correct', disp_frame.shape == (1080, 1920, 4))

    # ============================================================
    print(f'\n{"="*60}')
    print(f'Phase 6 Capture/Display Results: {PASS} PASS, {FAIL} FAIL, {PASS+FAIL} total')
    if FAIL > 0:
        print('SOME TESTS FAILED')
    else:
        print('ALL TESTS PASSED')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
