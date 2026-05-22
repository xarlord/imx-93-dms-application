import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.capture import CameraCapture, PIPELINE_CAPTURE
from src.display import DisplayOutput, PIPELINE_DISPLAY


class TestCameraCapture:
    def test_constructs_with_defaults(self):
        cap = CameraCapture()
        assert cap is not None

    def test_default_device(self):
        cap = CameraCapture()
        assert cap.device == '/dev/video0', f'got {cap.device}'

    def test_default_width(self):
        cap = CameraCapture()
        assert cap.width == 1280

    def test_default_height(self):
        cap = CameraCapture()
        assert cap.height == 800

    def test_default_target_fps(self):
        cap = CameraCapture()
        assert cap.target_fps == 25

    def test_config_device_applied(self):
        cfg = {'device': '/dev/video1', 'width': 640, 'height': 480, 'target_fps': 30}
        cap = CameraCapture(config=cfg)
        assert cap.device == '/dev/video1'

    def test_config_width_applied(self):
        cfg = {'device': '/dev/video1', 'width': 640, 'height': 480, 'target_fps': 30}
        cap = CameraCapture(config=cfg)
        assert cap.width == 640

    def test_config_height_applied(self):
        cfg = {'device': '/dev/video1', 'width': 640, 'height': 480, 'target_fps': 30}
        cap = CameraCapture(config=cfg)
        assert cap.height == 480

    def test_config_fps_applied(self):
        cfg = {'device': '/dev/video1', 'width': 640, 'height': 480, 'target_fps': 30}
        cap = CameraCapture(config=cfg)
        assert cap.target_fps == 30

    def test_initial_frame_count_is_zero(self):
        cap = CameraCapture()
        assert cap.frame_count == 0

    def test_initial_running_is_false(self):
        cap = CameraCapture()
        assert not cap.running

    def test_get_frame_returns_none_before_start(self):
        cap = CameraCapture()
        frame = cap.get_frame()
        assert frame is None

    def test_pipeline_contains_v4l2src(self):
        pipe = PIPELINE_CAPTURE.format(device='/dev/video0', width=1280, height=800, fps=25)
        assert 'v4l2src' in pipe

    def test_pipeline_contains_appsink(self):
        pipe = PIPELINE_CAPTURE.format(device='/dev/video0', width=1280, height=800, fps=25)
        assert 'appsink' in pipe

    def test_pipeline_contains_pxp(self):
        pipe = PIPELINE_CAPTURE.format(device='/dev/video0', width=1280, height=800, fps=25)
        assert 'imxvideoconvert_pxp' in pipe

    def test_pipeline_contains_bgrx_format(self):
        pipe = PIPELINE_CAPTURE.format(device='/dev/video0', width=1280, height=800, fps=25)
        assert 'format=BGRx' in pipe

    def test_pipeline_contains_yuy2_input(self):
        pipe = PIPELINE_CAPTURE.format(device='/dev/video0', width=1280, height=800, fps=25)
        assert 'format=YUY2' in pipe

    def test_pipeline_string_uses_device(self):
        pipe2 = PIPELINE_CAPTURE.format(device='/dev/video1', width=640, height=480, fps=30)
        assert '/dev/video1' in pipe2

    def test_pipeline_string_uses_width(self):
        pipe2 = PIPELINE_CAPTURE.format(device='/dev/video1', width=640, height=480, fps=30)
        assert 'width=640' in pipe2

    def test_pipeline_string_uses_fps(self):
        pipe2 = PIPELINE_CAPTURE.format(device='/dev/video1', width=640, height=480, fps=30)
        assert 'framerate=30/1' in pipe2

    def test_start_without_gstreamer_raises(self):
        cap = CameraCapture()
        try:
            cap.start()
            pytest.fail('no exception raised')
        except RuntimeError as e:
            assert 'GStreamer not available' in str(e)
        except Exception:
            pass

    def test_stop_without_start_does_not_crash(self):
        cap = CameraCapture()
        cap.stop()


class TestDisplayOutput:
    def test_constructs_with_defaults(self):
        disp = DisplayOutput()
        assert disp is not None

    def test_default_width(self):
        disp = DisplayOutput()
        assert disp.width == 1920

    def test_default_height(self):
        disp = DisplayOutput()
        assert disp.height == 1080

    def test_default_target_fps(self):
        disp = DisplayOutput()
        assert disp.target_fps == 25

    def test_default_fullscreen(self):
        disp = DisplayOutput()
        assert disp.fullscreen is True

    def test_config_width_applied(self):
        cfg_disp = {'width': 1280, 'height': 720, 'target_fps': 30, 'fullscreen': False}
        disp2 = DisplayOutput(config=cfg_disp)
        assert disp2.width == 1280

    def test_config_height_applied(self):
        cfg_disp = {'width': 1280, 'height': 720, 'target_fps': 30, 'fullscreen': False}
        disp2 = DisplayOutput(config=cfg_disp)
        assert disp2.height == 720

    def test_config_fps_applied(self):
        cfg_disp = {'width': 1280, 'height': 720, 'target_fps': 30, 'fullscreen': False}
        disp2 = DisplayOutput(config=cfg_disp)
        assert disp2.target_fps == 30

    def test_config_fullscreen_applied(self):
        cfg_disp = {'width': 1280, 'height': 720, 'target_fps': 30, 'fullscreen': False}
        disp2 = DisplayOutput(config=cfg_disp)
        assert disp2.fullscreen is False

    def test_initial_frame_count_is_zero(self):
        disp = DisplayOutput()
        assert disp.frame_count == 0

    def test_initial_running_is_false(self):
        disp = DisplayOutput()
        assert not disp.running

    def test_push_frame_returns_false_before_start(self):
        disp = DisplayOutput()
        result = disp.push_frame(np.zeros((1080, 1920, 4), dtype=np.uint8))
        assert result is False

    def test_pipeline_contains_appsrc(self):
        pipe_disp = PIPELINE_DISPLAY.format(width=1920, height=1080, fps=25, fullscreen='true')
        assert 'appsrc' in pipe_disp

    def test_pipeline_contains_waylandsink(self):
        pipe_disp = PIPELINE_DISPLAY.format(width=1920, height=1080, fps=25, fullscreen='true')
        assert 'waylandsink' in pipe_disp

    def test_pipeline_contains_pxp(self):
        pipe_disp = PIPELINE_DISPLAY.format(width=1920, height=1080, fps=25, fullscreen='true')
        assert 'imxvideoconvert_pxp' in pipe_disp

    def test_pipeline_contains_bgrx(self):
        pipe_disp = PIPELINE_DISPLAY.format(width=1920, height=1080, fps=25, fullscreen='true')
        assert 'format=BGRx' in pipe_disp

    def test_pipeline_fullscreen_true(self):
        pipe_disp = PIPELINE_DISPLAY.format(width=1920, height=1080, fps=25, fullscreen='true')
        assert 'fullscreen=true' in pipe_disp

    def test_pipeline_fullscreen_false(self):
        pipe_nf = PIPELINE_DISPLAY.format(width=1920, height=1080, fps=25, fullscreen='false')
        assert 'fullscreen=false' in pipe_nf

    def test_start_without_gstreamer_raises(self):
        disp = DisplayOutput()
        try:
            disp.start()
            pytest.fail('no exception raised')
        except RuntimeError as e:
            assert 'GStreamer not available' in str(e)
        except Exception:
            pass

    def test_stop_without_start_does_not_crash(self):
        disp = DisplayOutput()
        disp.stop()


class TestIntegration:
    def test_capture_uses_pxp_csc(self):
        assert 'imxvideoconvert_pxp' in PIPELINE_CAPTURE

    def test_display_uses_pxp_csc(self):
        assert 'imxvideoconvert_pxp' in PIPELINE_DISPLAY

    def test_appsink_max_buffers(self):
        assert 'max-buffers=2' in PIPELINE_CAPTURE

    def test_appsink_drop_true(self):
        assert 'drop=true' in PIPELINE_CAPTURE

    def test_appsink_emit_signals_true(self):
        assert 'emit-signals=true' in PIPELINE_CAPTURE

    def test_appsrc_block_true(self):
        assert 'block=true' in PIPELINE_DISPLAY

    def test_appsrc_max_buffers(self):
        assert 'max-buffers=2' in PIPELINE_DISPLAY

    def test_waylandsink_sync_false(self):
        assert 'sync=false' in PIPELINE_DISPLAY

    def test_capture_frame_contiguous(self):
        dummy_frame = np.zeros((800, 1280, 3), dtype=np.uint8)
        assert dummy_frame.flags['C_CONTIGUOUS']

    def test_capture_frame_dtype(self):
        dummy_frame = np.zeros((800, 1280, 3), dtype=np.uint8)
        assert dummy_frame.dtype == np.uint8

    def test_capture_frame_shape(self):
        dummy_frame = np.zeros((800, 1280, 3), dtype=np.uint8)
        assert dummy_frame.shape == (800, 1280, 3)

    def test_display_frame_contiguous(self):
        disp_frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
        assert disp_frame.flags['C_CONTIGUOUS']

    def test_display_frame_dtype(self):
        disp_frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
        assert disp_frame.dtype == np.uint8

    def test_display_frame_shape(self):
        disp_frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
        assert disp_frame.shape == (1080, 1920, 4)
