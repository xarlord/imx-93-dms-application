import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from src.utils.config import DMSConfig
from src.pipeline import DMSPipeline


@pytest.fixture
def dms_config_path():
    return os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')


@pytest.fixture
def dms_config(dms_config_path):
    return DMSConfig(dms_config_path)


@pytest.fixture
def mock_config():
    cfg = DMSConfig(None)
    cfg.set('models.mock', True)
    cfg.set('models.head_detector', '')
    cfg.set('models.landmark', '')
    cfg.set('models.iris', '')
    return cfg


@pytest.fixture
def sample_frame():
    return np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)


@pytest.fixture
def small_frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def black_frame():
    return np.zeros((800, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_face_bbox():
    return [540, 275, 740, 525]


@pytest.fixture
def sample_landmarks():
    landmarks = [(640, 400)]
    landmarks += [(580, 360), (600, 360)]
    landmarks += [(680, 360), (700, 360)]
    landmarks += [(570, 370), (580, 362), (595, 362), (600, 370), (595, 378), (580, 378)]
    landmarks += [(710, 370), (700, 362), (685, 362), (680, 370), (685, 378), (700, 378)]
    landmarks += [(610, 430), (625, 425), (640, 424), (655, 425),
                  (670, 430), (655, 436), (640, 437), (625, 436)]
    assert len(landmarks) == 25
    return landmarks


@pytest.fixture
def sample_left_eye_lm():
    return [(570, 370), (580, 362), (595, 362), (600, 370), (595, 378), (580, 378)]


@pytest.fixture
def sample_right_eye_lm():
    return [(710, 370), (700, 362), (685, 362), (680, 370), (685, 378), (700, 378)]


@pytest.fixture
def mock_pipeline(mock_config):
    return DMSPipeline(mock_config)


@pytest.fixture
def sample_results():
    return {
        'face_detected': False,
        'head_bbox': None,
        'landmarks': None,
        'left_iris': None,
        'right_iris': None,
        'ear_left': 0.0,
        'ear_right': 0.0,
        'ear': 0.0,
        'mar': 0.0,
        'kss': 1,
        'health_score': 100.0,
        'perclos': 0.0,
        'perclos_severity': 'none',
        'blink_rate': 0.0,
        'yawn_active': False,
        'microsleep_state': 'open',
        'gaze_yaw': 0.0,
        'gaze_pitch': 0.0,
        'zone_id': 15,
        'zone_name': 'UNKNOWN',
        'is_on_road': False,
        'is_distracted': False,
        'warning_level': 'none',
        'final_yaw': 0.0,
        'final_pitch': 0.0,
        'proj_x': 960,
        'proj_y': 540,
        'continuous_offroad': 0.0,
        'gates_closed': True,
    }
