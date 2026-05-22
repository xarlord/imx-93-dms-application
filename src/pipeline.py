"""Pipeline orchestrator: connects all DMS stages in a synchronous pipeline."""
import logging
import os
import time
from collections import deque
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from .head_detector import HeadDetector
from .landmark_detector import LandmarkDetector
from .iris_detector import IrisDetector
from .gaze_estimator import GazeEstimator
from .behavioral.perclos import PERCLOSDetector
from .behavioral.blink import BlinkDetector
from .behavioral.yawn import YawnDetector
from .behavioral.microsleep import MicrosleepDetector
from .behavioral.distraction import DistractionDetector
from .behavioral.drowsiness import DrowsinessScorer
from .behavioral.warning import WarningManager
from .dashboard.renderer import DashboardRenderer
from .dashboard.overlays import draw_landmarks, draw_iris, draw_gaze_ray, draw_head_bbox
from .dashboard.warnings import WarningVisuals
from .utils.geometry import compute_ear, compute_mar, compute_head_pose
from .utils.image import crop_with_padding, resize_for_model
from .utils.zone_loader import ZoneLoader

logger = logging.getLogger('dms.pipeline')


class DMSPipeline:
    """Orchestrates the full DMS inference pipeline.

    Stages per frame:
    1. YOLO head detection
    2. Head crop + resize
    3. Landmark detection (25 points)
    4. Eye crop + resize
    5. Iris detection (both eyes)
    6. Gaze estimation
    7. Behavioral analysis
    8. Dashboard rendering

    Synchronous pipeline: process_frame() runs the full chain on the calling thread.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.fps: int = config.get('camera.target_fps', 25)
        self.mock: bool = config.get('models.mock', False)
        self._config_dir: str | None = self._find_config_dir(config)
        if self._config_dir is None:
            raise RuntimeError(
                'Pipeline requires config._config_path for relative-path resolution. '
                'Pass a DMSConfig loaded from a YAML file.')
        self._init_models(config)
        self._init_behavioral(config)
        self._init_dashboard(config)
        self._init_state(config)

    def _init_models(self, config: Any) -> None:
        """Instantiate detection and estimation models from config.

        Loads HeadDetector, LandmarkDetector, IrisDetector, GazeEstimator,
        and optional ZoneLoader for gaze zone classification.

        Args:
            config: ``DMSConfig`` providing model paths and parameters.
        """
        model_dir: str = config.get('models.head_detector', '')
        self.head_det = HeadDetector(
            model_path=model_dir if not self.mock else None,
            use_npu=config.get('models.use_npu', True),
            mock=self.mock,
        )

        landmark_path: str = config.get('models.landmark', '')
        self.landmark_det = LandmarkDetector(
            model_path=landmark_path if not self.mock else None,
            use_npu=config.get('models.use_npu', True),
            mock=self.mock,
        )

        iris_path: str = config.get('models.iris', '')
        self.iris_det = IrisDetector(
            model_path=iris_path if not self.mock else None,
            use_npu=config.get('models.use_npu', True),
            mock=self.mock,
        )

        self.gaze_est = GazeEstimator(
            yaw_weight=config.get('gaze.yaw_weight', 32.0),
            pitch_weight=config.get('gaze.pitch_weight', 45.0),
            ppd_yaw=config.get('gaze.ppd_yaw', 5.0),
            ppd_pitch=config.get('gaze.ppd_pitch', 5.0),
            display_w=config.get('display.width', 1920),
            display_h=config.get('display.height', 1080),
            alpha_fixation=config.get('gaze.stabilization_alpha_fixation', 0.3),
            alpha_saccade=config.get('gaze.stabilization_alpha_saccade', 0.8),
            saccade_velocity=config.get('gaze.saccade_velocity_threshold', 15.0),
        )
        zones_path: str | None = config.get('gaze.zones_file')
        if zones_path and not os.path.isabs(zones_path):
            zones_path = os.path.join(self._config_dir, zones_path)
        if zones_path:
            try:
                zl = ZoneLoader(zones_path)
                self.gaze_est.set_zone_loader(zl)
                logger.info('Zone loader wired: %d zones from %s', len(zl.zones), zones_path)
            except (OSError, ValueError, KeyError) as e:
                logger.warning('Failed to load zones from %s: %s', zones_path, e)

    def _init_behavioral(self, config: Any) -> None:
        """Instantiate all behavioral analyzers from config.

        Creates PERCLOS, blink, yawn, microsleep, distraction, drowsiness,
        and warning escalation modules.

        Args:
            config: ``DMSConfig`` providing thresholds and timing parameters.
        """
        self.perclos = PERCLOSDetector(window_sec=config.get('thresholds.perclos_window_sec', 60))
        self.blink = BlinkDetector()
        self.yawn = YawnDetector(
            mar_threshold=config.get('thresholds.yawn_mar', 0.5),
            min_duration=config.get('thresholds.yawn_min_duration', 0.5),
        )
        self.microsleep = MicrosleepDetector(
            ear_close_threshold=config.get('thresholds.ear_close', 0.10),
        )
        self.distraction = DistractionDetector(
            saccade_tolerance_ms=config.get('timing.saccade_tolerance_ms', 50),
            vats_window_sec=config.get('timing.vats_window_sec', 30),
            vats_threshold_sec=config.get('timing.vats_threshold_sec', 10.0),
            fps=config.get('camera.target_fps', 25),
            addw_activation_speed=config.get('vehicle.addw_activation_speed', 20),
        )
        self.drowsiness = DrowsinessScorer(
            learning_period_sec=config.get('calibration.learning_period_sec', 600),
        )
        self.warning = WarningManager(
            advisory_sec=config.get('warnings.escalation_advisory_sec', 3.0),
            escalating_sec=config.get('warnings.escalation_escalating_sec', 5.0),
            intervention_sec=config.get('warnings.escalation_intervention_sec', 5.0),
            ddaw_activation_speed=config.get('vehicle.ddaw_activation_speed', 70),
            addw_activation_speed=config.get('vehicle.addw_activation_speed', 20),
            acoustic_advisory=config.get('warnings.acoustic_file_advisory'),
            acoustic_escalating=config.get('warnings.acoustic_file_escalating'),
            acoustic_intervention=config.get('warnings.acoustic_file_intervention'),
            acoustic_emergency=config.get('warnings.acoustic_file_emergency'),
            config_dir=self._config_dir,
        )

    def _init_dashboard(self, config: Any) -> None:
        """Instantiate the dashboard renderer and warning visuals.

        Args:
            config: ``DMSConfig`` providing display dimensions.
        """
        self.renderer = DashboardRenderer(
            width=config.get('display.width', 1920),
            height=config.get('display.height', 1080),
        )
        self.warning_vis = WarningVisuals()

    def _init_state(self, config: Any) -> None:
        """Initialize runtime state: FPS tracking, vehicle speed, no-face timer.

        Args:
            config: ``DMSConfig`` providing vehicle and camera settings.
        """
        self._frame_times: deque[float] = deque(maxlen=100)
        self._current_fps: float = 0.0
        self._no_face_since: float | None = None
        self._no_face_kss_peak: int = 1
        self._no_face_peak_time: float | None = None

        self._vehicle_speed: int = config.get('vehicle.simulated_speed', 0)
        self._last_gate_log_time: float = 0.0
        if self._vehicle_speed == 0:
            logger.warning('Vehicle speed is 0 — DDAW/ADDW gates closed. '
                           'Configure vehicle.simulated_speed or wire CAN bus.')

    @staticmethod
    def _find_config_dir(config: Any) -> str | None:
        if getattr(config, 'config_path', None) is not None:
            return os.path.dirname(config.config_path)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')

    def process_frame(self, frame: NDArray[np.uint8], timestamp: float | None = None) -> dict[str, Any]:
        """Process a single frame through the full pipeline (synchronous).

        Args:
            frame: numpy [H, W, 3 or 4] uint8 BGR or BGRx frame.
            timestamp: optional monotonic timestamp.

        Returns:
            dict with all pipeline results.
        """
        t0: float = time.monotonic()
        now: float = timestamp or t0
        dt: float = 1.0 / max(self.fps, 1)

        results: dict[str, Any] = self._init_results()

        if self.gates_closed and (now - self._last_gate_log_time) > 60.0:
            logger.info('DDAW/ADDW gates still closed (vehicle_speed=0). '
                        'Frame processing continues but no warnings will fire.')
            self._last_gate_log_time = now

        face_ok: bool = self._run_detection_stages(frame, results, now, dt)
        if face_ok:
            self._run_analysis_stages(frame, results, now, dt)

        elapsed: float = time.monotonic() - t0
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 10:
            avg: float = sum(self._frame_times) / len(self._frame_times)
            self._current_fps = 1.0 / avg if avg > 0 else 0

        return results

    def _init_results(self) -> dict[str, Any]:
        """Create a results dict with default values for all pipeline outputs.

        Returns:
            Dict with keys for every pipeline stage output, pre-populated
            with safe defaults (no face detected, KSS 1, etc.).
        """
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
            'gates_closed': self.gates_closed,
        }

    def _run_detection_stages(self, frame: NDArray[np.uint8], results: dict[str, Any], now: float, dt: float) -> bool:
        """Run head detection through gaze estimation, filling *results*.

        Args:
            frame: RGB camera frame ``[H, W, 3]``.
            results: Mutable results dict to update in place.
            now: Monotonic timestamp of the current frame.
            dt: Frame delta time in seconds.

        Returns:
            ``True`` if a face was detected and landmarks extracted;
            ``False`` otherwise.
        """
        best: dict[str, Any] | None = self.head_det.get_best_detection(frame)
        if best is None:
            self._update_behavioral_no_face(results, now, dt)
            results['warning_level'] = self.warning.level
            return False

        results['face_detected'] = True
        if (self._no_face_peak_time is not None and
                (now - self._no_face_peak_time) > 0.5):
            self._no_face_kss_peak = 1
            self._no_face_peak_time = None
        self._no_face_since = None
        bbox: list[float] = best['bbox']
        results['head_bbox'] = bbox
        results['head_confidence'] = best.get('confidence', 0)

        head_crop: NDArray[np.uint8]
        crop_info: dict[str, Any]
        head_crop, crop_info = crop_with_padding(frame, bbox, padding_ratio=0.20)

        landmarks: list[tuple[float, float]] | None = self.landmark_det.detect(head_crop, bbox)
        if landmarks is None:
            self._update_behavioral_no_face(results, now, dt)
            results['warning_level'] = self.warning.level
            return False

        results['landmarks'] = landmarks

        left_eye_lm: list[tuple[float, float]] = [landmarks[i] for i in range(5, 11)]
        right_eye_lm: list[tuple[float, float]] = [landmarks[i] for i in range(11, 17)]
        left_iris: tuple[float, float, float] | None
        right_iris: tuple[float, float, float] | None
        left_iris, right_iris = self.iris_det.detect(frame, left_eye_lm, right_eye_lm)
        results['left_iris'] = left_iris
        results['right_iris'] = right_iris

        ear_left: float = compute_ear(left_eye_lm)
        ear_right: float = compute_ear(right_eye_lm)
        ear_avg: float = (ear_left + ear_right) / 2.0
        results['ear_left'] = ear_left
        results['ear_right'] = ear_right
        results['ear'] = ear_avg

        mouth_lm: list[tuple[float, float]] = [landmarks[i] for i in range(17, 25)]
        mar: float = compute_mar(mouth_lm)
        results['mar'] = mar

        face_bbox: list[float] = [bbox[0], bbox[1], bbox[2], bbox[3]]
        head_yaw: float
        head_pitch: float
        head_yaw, head_pitch = compute_head_pose(landmarks, face_bbox)

        gaze: dict[str, Any] = self.gaze_est.estimate(
            landmarks, left_iris, right_iris,
            face_bbox, dt=dt,
        )
        results.update(gaze)

        self._current_head_yaw: float = head_yaw
        self._current_head_pitch: float = head_pitch
        return True

    def _run_analysis_stages(self, frame: NDArray[np.uint8], results: dict[str, Any], now: float, dt: float) -> None:
        """Run behavioral analyzers on detection results.

        Args:
            frame: RGB camera frame ``[H, W, 3]``.
            results: Mutable results dict already populated by detection.
            now: Monotonic timestamp of the current frame.
            dt: Frame delta time in seconds.
        """
        self._update_behavioral(
            results, results['ear_left'], results['ear_right'],
            results['mar'], self._current_head_yaw, self._current_head_pitch,
            now, dt)

    def render_dashboard(self, frame_bgrx: NDArray[np.uint8], results: dict[str, Any], fps: float | None = None) -> NDArray[np.uint8]:
        """Render the dashboard frame with overlays.

        Args:
            frame_bgrx: camera frame [H, W, 4] BGRx.
            results: pipeline results dict.
            fps: display FPS.

        Returns:
            BGRx [H, W, 4] numpy array.
        """
        display_fps: float = fps or self._current_fps
        dashboard: NDArray[np.uint8] = self.renderer.render(frame_bgrx, results, fps=display_fps)

        if results.get('head_bbox'):
            draw_head_bbox(dashboard, results['head_bbox'],
                           confidence=results.get('head_confidence', 0),
                           scale_x=1.0, scale_y=1.0)

        if results.get('landmarks'):
            draw_landmarks(dashboard, results['landmarks'],
                           scale_x=1.0, scale_y=1.0)

        draw_iris(dashboard, results.get('left_iris'),
                  results.get('right_iris'),
                  scale_x=1.0, scale_y=1.0)

        if results.get('landmarks') and results['landmarks'][0] != (0, 0):
            draw_gaze_ray(dashboard, results['landmarks'][0],
                          results.get('final_yaw', 0),
                          results.get('final_pitch', 0),
                          scale_x=1.0, scale_y=1.0)

        return dashboard

    @property
    def current_fps(self) -> float:
        return self._current_fps

    @property
    def gates_closed(self) -> bool:
        """True when DDAW/ADDW gates are inactive due to vehicle_speed == 0."""
        return self._vehicle_speed == 0

    def stop(self) -> None:
        """Stop background threads (audio worker). Call on pipeline shutdown."""
        self.warning.stop()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any | None) -> bool:
        self.stop()
        return False

    def _update_behavioral(self, results: dict[str, Any], ear_left: float, ear_right: float, mar: float,
                           head_yaw: float, head_pitch: float, now: float, dt: float) -> None:
        """Run all behavioral analyzers and update results.

        Args:
            results: Mutable results dict to update in place.
            ear_left: Left eye aspect ratio.
            ear_right: Right eye aspect ratio.
            mar: Mouth aspect ratio.
            head_yaw: Head yaw in degrees.
            head_pitch: Head pitch in degrees.
            now: Monotonic timestamp.
            dt: Frame delta time in seconds.
        """
        self.perclos.update(ear_left, ear_right)
        results['perclos'] = self.perclos.perclos()
        results['perclos_severity'] = self.perclos.severity()

        # Blink
        self.blink.update(ear_left, ear_right)
        results['blink_rate'] = self.blink.blink_rate()

        # Yawn
        self.yawn.update(mar)
        results['yawn_active'] = self.yawn.is_yawning

        # Microsleep
        ms_state: str = self.microsleep.update(ear_left, ear_right)
        results['microsleep_state'] = ms_state

        # Distraction
        zone_id: int = results.get('zone_id', 15)
        is_on_road: bool = results.get('is_on_road', False)
        gaze_confidence: float = results.get('gaze_confidence', 1.0)
        dist: dict[str, Any] = self.distraction.update(is_on_road, self._vehicle_speed, dt,
                                       gaze_confidence=gaze_confidence)
        results['is_distracted'] = dist.get('addw_nominal', False)
        results['continuous_offroad'] = dist.get('continuous_offroad', 0)

        # Drowsiness score
        drowsiness_result: dict[str, Any] = self.drowsiness.update(
            results['perclos_severity'],
            results['blink_rate'],
            results['yawn_active'],
            results['microsleep_state'],
            head_yaw=head_yaw,
            head_pitch=head_pitch,
        )
        kss: int = drowsiness_result['kss']
        results['kss'] = kss
        results['health_score'] = drowsiness_result['health_score']

        # Warning escalation
        self.warning.update(
            kss, dist, ms_state,
            vehicle_speed=self._vehicle_speed,
        )
        results['warning_level'] = self.warning.level

    def _update_behavioral_no_face(self, results: dict[str, Any], now: float, dt: float) -> None:
        """Update behavioral state when no face is detected.

        Per Euro NCAP Driver Engagement v0.9, sustained no-face should
        escalate toward 'unresponsive driver' category.
        Graded KSS ramp: 3s -> KSS 6, 5s -> KSS 8, 6s -> KSS 9.
        Latched to highest tier: brief face re-detection doesn't reset.
        Decays after 500ms of sustained face re-detection.
        """
        # Distraction: treat no face as off-road
        dist: dict[str, Any] = self.distraction.update(False, self._vehicle_speed, dt,
                                       gaze_confidence=0.0)
        results['continuous_offroad'] = dist.get('continuous_offroad', 0)
        results['is_distracted'] = dist.get('addw_nominal', False)

        # Graded no-face KSS escalation with latching
        if self._no_face_since is None:
            self._no_face_since = now
        no_face_duration: float = now - self._no_face_since
        no_face_kss: int
        if no_face_duration >= 6.0:
            no_face_kss = 9
        elif no_face_duration >= 5.0:
            no_face_kss = 8
        elif no_face_duration >= 3.0:
            no_face_kss = 6
        else:
            no_face_kss = 1

        # Latch: never de-escalate during a no-face episode
        if no_face_kss > self._no_face_kss_peak:
            self._no_face_kss_peak = no_face_kss
            self._no_face_peak_time = now
        no_face_kss = max(no_face_kss, self._no_face_kss_peak)

        results['no_face_kss_latched'] = no_face_kss

        # Warning (pass vehicle_speed for regulatory gating)
        self.warning.update(no_face_kss, dist, 'open',
                            vehicle_speed=self._vehicle_speed)
        results['warning_level'] = self.warning.level
