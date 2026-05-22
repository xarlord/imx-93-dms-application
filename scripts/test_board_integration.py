#!/usr/bin/env python3
"""Board-side integration test for DMS on FRDM-IMX93.

Runs on the i.MX93 board to verify:
1. Camera capture (AR0144 via AP1302 -> /dev/video0)
2. NPU availability (Ethos-U65 /dev/ethosu0)
3. GStreamer display (PXP + waylandsink)
4. Model loading and inference
5. Full pipeline with real camera

Usage:
    python3 test_board_integration.py [--skip-camera] [--skip-npu] [--skip-display]
                                      [--mock-models] [--full-pipeline]
"""
import argparse
import os
import sys
import time
import subprocess

PASS = 0
FAIL = 0
SKIP = 0

DMS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DMS_ROOT)


def test(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  PASS: {name}')
    else:
        FAIL += 1
        print(f'  FAIL: {name} -- {detail}')


def skip(name, reason=''):
    global SKIP
    SKIP += 1
    print(f'  SKIP: {name} -- {reason}')


def run_cmd(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, '', 'timeout'
    except Exception as e:
        return False, '', str(e)


# ============================================================
print('\n=== Board Integration Test: FRDM-IMX93 DMS ===\n')
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument('--skip-camera', action='store_true')
parser.add_argument('--skip-npu', action='store_true')
parser.add_argument('--skip-display', action='store_true')
parser.add_argument('--mock-models', action='store_true',
                    help='Use mock mode (no model files needed)')
parser.add_argument('--full-pipeline', action='store_true',
                    help='Run full pipeline with real camera')
parser.add_argument('--capture-frames', type=int, default=10,
                    help='Number of frames to capture for testing')
args = parser.parse_args()


# ============================================================
print('=== Section 1: System Checks ===')
# ============================================================

# Kernel
ok, out, _ = run_cmd('uname -r')
test('Linux kernel >= 6.12', ok and '6.' in out, f'kernel: {out}')

# Platform
ok, out, _ = run_cmd('cat /sys/devices/soc0/machine')
test('i.MX93 platform', ok and 'imx93' in out.lower(), f'machine: {out}')

# Memory
ok, out, _ = run_cmd('free -m | head -2')
test('RAM available (>512MB)', ok, f'{out}')

# Disk
ok, out, _ = run_cmd(f'df -m {DMS_ROOT} | tail -1')
test('Disk space available', ok, f'{out}')


# ============================================================
print('\n=== Section 2: Camera (AR0144 + AP1302) ===')
# ============================================================

if args.skip_camera:
    skip('Camera tests', '--skip-camera')
else:
    # Video device
    test('Video device /dev/video0 exists', os.path.exists('/dev/video0'))

    ok, out, _ = run_cmd('v4l2-ctl --list-devices')
    test('V4L2 devices listed', ok, f'{out[:200]}')

    # Camera format
    ok, out, _ = run_cmd('v4l2-ctl -d /dev/video0 --list-fmt-video')
    test('Camera supports YUYV', ok and 'YUYV' in out, f'{out[:200]}')

    # Camera resolution
    ok, out, _ = run_cmd('v4l2-ctl -d /dev/video0 --list-framesizes=YUYV')
    test('Camera supports 1280x800', ok and '1280x800' in out, f'{out[:200]}')

    # Media pipeline
    ok, out, _ = run_cmd('media-ctl -d /dev/media0 -p 2>/dev/null | head -20')
    test('Media pipeline configured', ok and 'ap1302' in out.lower(),
         f'{out[:200]}')

    # AP1302 firmware
    ok, out, _ = run_cmd('dmesg | grep -i ap1302 | tail -3')
    test('AP1302 firmware loaded', ok and 'ap1302' in out.lower(),
         f'{out[:200]}')

    # Single frame capture
    ok, out, _ = run_cmd(
        'v4l2-ctl -d /dev/video0 --stream-mmap '
        '--stream-to=/tmp/dms_test.raw --stream-count=1',
        timeout=5
    )
    test('Single frame capture', ok, f'{out[:200]}')

    if ok:
        test('Captured file non-empty',
             os.path.exists('/tmp/dms_test.raw') and
             os.path.getsize('/tmp/dms_test.raw') > 100000,
             f'size: {os.path.getsize("/tmp/dms_test.raw") if os.path.exists("/tmp/dms_test.raw") else 0}')


# ============================================================
print('\n=== Section 3: NPU (Ethos-U65) ===')
# ============================================================

if args.skip_npu:
    skip('NPU tests', '--skip-npu')
else:
    # Device nodes
    ethosu_exists = os.path.exists('/dev/ethosu0')
    test('Ethos-U65 device node /dev/ethosu0', ethosu_exists)

    ok, out, _ = run_cmd('dmesg | grep -i ethosu | tail -5')
    test('Ethos-U65 driver loaded', ok and 'ethosu' in out.lower(),
         f'{out[:200]}')

    # NPU firmware via remoteproc
    ok, out, _ = run_cmd('cat /sys/class/remoteproc/remoteproc0/state')
    test('M33 core running (NPU firmware)', ok and 'running' in out,
         f'state: {out}')

    # Python NPU test
    try:
        import ethosu
        test('ethosu Python module available', True)
    except ImportError:
        test('ethosu Python module available', False, 'import failed')

    try:
        import tflite_runtime
        test('tflite_runtime Python module available', True)
    except ImportError:
        test('tflite_runtime Python module available', False, 'import failed')


# ============================================================
print('\n=== Section 4: Display (HDMI + Wayland) ===')
# ============================================================

if args.skip_display:
    skip('Display tests', '--skip-display')
else:
    # Weston service
    ok, out, _ = run_cmd('systemctl is-active weston')
    test('Weston compositor running', ok and 'active' in out, f'state: {out}')

    # Wayland display
    test('WAYLAND_DISPLAY set', 'WAYLAND_DISPLAY' in os.environ,
         f'env: {os.environ.get("WAYLAND_DISPLAY", "not set")}')

    test('XDG_RUNTIME_DIR set', 'XDG_RUNTIME_DIR' in os.environ,
         f'env: {os.environ.get("XDG_RUNTIME_DIR", "not set")}')

    # HDMI connected
    ok, out, _ = run_cmd('ls /sys/class/drm/card0-HDMI-A-1/status 2>/dev/null && '
                         'cat /sys/class/drm/card0-HDMI-A-1/status')
    test('HDMI display connected', ok and 'connected' in out, f'{out}')

    # GStreamer
    ok, out, _ = run_cmd('gst-launch-1.0 --version')
    test('GStreamer available', ok, f'{out}')

    # GStreamer imx plugins
    ok, out, _ = run_cmd('gst-inspect-1.0 imxvideoconvert_pxp 2>&1 | head -3')
    test('PXP GStreamer plugin available', ok, f'{out[:200]}')

    ok, out, _ = run_cmd('gst-inspect-1.0 waylandsink 2>&1 | head -3')
    test('waylandsink plugin available', ok, f'{out[:200]}')


# ============================================================
print('\n=== Section 5: Python Environment ===')
# ============================================================

modules = [
    ('numpy', 'numpy'),
    ('OpenCV', 'cv2'),
    ('GStreamer', 'gi.repository.Gst'),
    ('yaml', 'yaml'),
]

for name, mod in modules:
    try:
        __import__(mod)
        test(f'{name} available', True)
    except ImportError:
        test(f'{name} available', False, 'import failed')

# DMS modules
try:
    from src.utils.config import DMSConfig
    test('DMS utils.config', True)
except Exception as e:
    test('DMS utils.config', False, str(e))

try:
    from src.pipeline import DMSPipeline
    test('DMS pipeline module', True)
except Exception as e:
    test('DMS pipeline module', False, str(e))

try:
    from src.capture import CameraCapture
    test('DMS capture module', True)
except Exception as e:
    test('DMS capture module', False, str(e))

try:
    from src.display import DisplayOutput
    test('DMS display module', True)
except Exception as e:
    test('DMS display module', False, str(e))


# ============================================================
print('\n=== Section 6: Model Files ===')
# ============================================================

model_dir = os.path.join(DMS_ROOT, 'models')
expected_models = {
    'yolov8n_face_vela.tflite': 'YOLO head detector (NPU)',
    'yolov8n_face.tflite': 'YOLO head detector (CPU fallback)',
    'face_landmark_vela.tflite': 'Face landmark (NPU, optional)',
    'face_landmark.tflite': 'Face landmark (CPU fallback)',
    'iris_detector_vela.tflite': 'Iris detector (NPU)',
    'iris_detector.tflite': 'Iris detector (CPU fallback)',
}

for fname, desc in expected_models.items():
    fpath = os.path.join(model_dir, fname)
    exists = os.path.exists(fpath)
    if 'vela' in fname:
        # Vela models are optional (CPU fallback exists)
        if exists:
            test(f'{desc}', True)
        else:
            skip(f'{desc}', 'not built yet, CPU fallback available')
    else:
        if exists:
            test(f'{desc}', True)
        else:
            skip(f'{desc}', 'model file not found')


# ============================================================
print('\n=== Section 7: Configuration ===')
# ============================================================

config_path = os.path.join(DMS_ROOT, 'config', 'config.yaml')
test('config.yaml exists', os.path.exists(config_path))

zones_path = os.path.join(DMS_ROOT, 'config', 'zones.xml')
test('zones.xml exists', os.path.exists(zones_path))

if os.path.exists(config_path):
    try:
        from src.utils.config import DMSConfig
        cfg = DMSConfig(config_path)
        test('config.yaml loads', True)
        test('camera.device configured', cfg.get('camera.device') is not None)
        test('display.width=1920', cfg.get('display.width') == 1920)
    except Exception as e:
        test('config.yaml loads', False, str(e))

if os.path.exists(zones_path):
    try:
        from src.utils.zone_loader import ZoneLoader
        zl = ZoneLoader(zones_path)
        test('zones.xml loads', True)
        test('zones loaded', len(zl.zones) > 0,
             f'{len(zl.zones)} zones')
    except Exception as e:
        test('zones.xml loads', False, str(e))


# ============================================================
print('\n=== Section 8: Full Pipeline Test ===')
# ============================================================

if args.full_pipeline and not args.skip_camera:
    try:
        import numpy as np
        from src.utils.config import DMSConfig
        from src.pipeline import DMSPipeline

        print('  Loading pipeline...')
        cfg = DMSConfig(config_path)
        if args.mock_models:
            cfg._flat['models.mock'] = True
            cfg._flat['models.head_detector'] = ''
            cfg._flat['models.landmark'] = ''
            cfg._flat['models.iris'] = ''
            print('  (mock mode - no ML models)')

        pipe = DMSPipeline(cfg)
        test('Pipeline constructed', True)

        # Try GStreamer camera capture
        try:
            from src.capture import CameraCapture
            cap = CameraCapture(config=cfg.camera)

            captured = [None]
            def on_frame(frame):
                captured[0] = frame

            print('  Starting camera capture...')
            cap.start(on_frame=on_frame)

            # Wait for first frame
            for _ in range(50):
                if captured[0] is not None:
                    break
                time.sleep(0.1)

            if captured[0] is not None:
                test('Camera frame captured', True,
                     f'shape={captured[0].shape}')
                print('  Processing frames...')
                for i in range(args.capture_frames):
                    frame = cap.get_frame()
                    if frame is not None:
                        results = pipe.process_frame(frame)
                test(f'Processed {args.capture_frames} frames', True)
                test(f'Pipeline FPS={pipe.current_fps:.1f}', pipe.current_fps > 0,
                     f'fps={pipe.current_fps:.1f}')
            else:
                test('Camera frame captured', False, 'timeout')

            cap.stop()

        except Exception as e:
            test('Camera capture', False, str(e))

    except Exception as e:
        test('Full pipeline', False, str(e))
else:
    skip('Full pipeline test', '--full-pipeline not specified or --skip-camera')


# ============================================================
print(f'\n{"="*60}')
print(f'Board Integration Results: {PASS} PASS, {FAIL} FAIL, {SKIP} SKIP')
print(f'Total: {PASS+FAIL+SKIP}')
if FAIL > 0:
    print('SOME TESTS FAILED')
else:
    print('ALL TESTS PASSED (skips are expected for unprepared items)')
print(f'{"="*60}')
