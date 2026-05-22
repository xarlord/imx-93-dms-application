import pytest
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import DMSConfig
from src.behavioral.perclos import PERCLOSDetector
from src.behavioral.blink import BlinkDetector
from src.behavioral.yawn import YawnDetector
from src.behavioral.microsleep import MicrosleepDetector
from src.behavioral.distraction import DistractionDetector
from src.behavioral.drowsiness import DrowsinessScorer
from src.behavioral.warning import WarningManager


class TestPERCLOS:
    def test_perclos_33_percent(self):
        pc = PERCLOSDetector(window_sec=60, close_threshold=0.10)
        t = time.monotonic()
        for i in range(60):
            ear_l = 0.05 if i < 20 else 0.30
            ear_r = 0.05 if i < 20 else 0.30
            t += 0.04
            pc.update(ear_l, ear_r, timestamp=t)
        perclos_val = pc.perclos()
        assert 0.30 < perclos_val < 0.36, f"got {perclos_val:.3f}"

    def test_perclos_severity_severe(self):
        pc = PERCLOSDetector(window_sec=60, close_threshold=0.10)
        t = time.monotonic()
        for i in range(60):
            ear_l = 0.05 if i < 20 else 0.30
            ear_r = 0.05 if i < 20 else 0.30
            t += 0.04
            pc.update(ear_l, ear_r, timestamp=t)
        assert pc.severity() == 'severe', f"got {pc.severity()}"

    def test_perclos_drops_after_open_frames(self):
        pc = PERCLOSDetector(window_sec=60, close_threshold=0.10)
        t = time.monotonic()
        for i in range(60):
            ear_l = 0.05 if i < 20 else 0.30
            ear_r = 0.05 if i < 20 else 0.30
            t += 0.04
            pc.update(ear_l, ear_r, timestamp=t)
        for i in range(500):
            t += 0.04
            pc.update(0.30, 0.30, timestamp=t)
        assert pc.perclos() < 0.10, f"got {pc.perclos():.3f}"

    def test_perclos_severity_none(self):
        pc = PERCLOSDetector(window_sec=60, close_threshold=0.10)
        t = time.monotonic()
        for i in range(60):
            ear_l = 0.05 if i < 20 else 0.30
            ear_r = 0.05 if i < 20 else 0.30
            t += 0.04
            pc.update(ear_l, ear_r, timestamp=t)
        for i in range(500):
            t += 0.04
            pc.update(0.30, 0.30, timestamp=t)
        assert pc.severity() == 'none'

    def test_perclos_mild(self):
        pc = PERCLOSDetector(window_sec=60, close_threshold=0.10)
        pc.reset()
        t = time.monotonic()
        for i in range(100):
            ear = 0.05 if i < 15 else 0.30
            t += 0.04
            pc.update(ear, ear, timestamp=t)
        assert pc.severity() == 'mild', f"got PERCLOS={pc.perclos():.3f} sev={pc.severity()}"


class TestBlink:
    def test_blink_detected(self):
        bd = BlinkDetector(blink_threshold=0.19, min_duration_ms=50)
        t = time.monotonic()
        for i in range(6):
            t += 0.04
            bd.update(0.05, 0.05, timestamp=t)
        for i in range(10):
            t += 0.04
            bd.update(0.30, 0.30, timestamp=t)
        assert bd.blink_count == 1, f"got {bd.blink_count}"

    def test_blink_rate(self):
        bd = BlinkDetector(blink_threshold=0.19, min_duration_ms=50)
        t = time.monotonic()
        for i in range(6):
            t += 0.04
            bd.update(0.05, 0.05, timestamp=t)
        for i in range(10):
            t += 0.04
            bd.update(0.30, 0.30, timestamp=t)
        assert abs(bd.blink_rate(window_sec=60, timestamp=t) - 1.0) < 0.1

    def test_short_blink_rejected(self):
        bd = BlinkDetector(blink_threshold=0.19, min_duration_ms=50)
        bd.reset()
        t = time.monotonic()
        t += 0.04
        bd.update(0.05, 0.05, timestamp=t)
        t += 0.04
        bd.update(0.30, 0.30, timestamp=t)
        assert bd.blink_count == 0, f"got {bd.blink_count}"


class TestYawn:
    def test_yawn_detected(self):
        yn = YawnDetector(mar_threshold=0.5, min_duration=0.5)
        t = time.monotonic()
        for i in range(20):
            t += 0.04
            yn.update(0.8, timestamp=t)
        assert yn.is_yawning

    def test_yawn_count_while_yawning(self):
        yn = YawnDetector(mar_threshold=0.5, min_duration=0.5)
        t = time.monotonic()
        for i in range(20):
            t += 0.04
            yn.update(0.8, timestamp=t)
        assert yn.yawn_count == 0

    def test_yawn_ended(self):
        yn = YawnDetector(mar_threshold=0.5, min_duration=0.5)
        t = time.monotonic()
        for i in range(20):
            t += 0.04
            yn.update(0.8, timestamp=t)
        for i in range(10):
            t += 0.04
            yn.update(0.2, timestamp=t)
        assert not yn.is_yawning

    def test_yawn_count_after_end(self):
        yn = YawnDetector(mar_threshold=0.5, min_duration=0.5)
        t = time.monotonic()
        for i in range(20):
            t += 0.04
            yn.update(0.8, timestamp=t)
        for i in range(10):
            t += 0.04
            yn.update(0.2, timestamp=t)
        assert yn.yawn_count == 1

    def test_brief_yawn_rejected(self):
        yn = YawnDetector(mar_threshold=0.5, min_duration=0.5)
        yn.reset()
        t = time.monotonic()
        for i in range(7):
            t += 0.04
            yn.update(0.8, timestamp=t)
        for i in range(5):
            t += 0.04
            yn.update(0.2, timestamp=t)
        assert yn.yawn_count == 0, f"got {yn.yawn_count}"


class TestMicrosleep:
    def test_microsleep_detected(self):
        ms = MicrosleepDetector(ear_close_threshold=0.10)
        t = time.monotonic()
        states = []
        for i in range(38):
            t += 0.04
            state = ms.update(0.05, 0.05, timestamp=t)
            states.append(state)
        assert 'microsleep' in states, f"states: {set(states)}"

    def test_sleep_detected(self):
        ms = MicrosleepDetector(ear_close_threshold=0.10)
        t = time.monotonic()
        states = []
        for i in range(38):
            t += 0.04
            state = ms.update(0.05, 0.05, timestamp=t)
            states.append(state)
        for i in range(38):
            t += 0.04
            state = ms.update(0.05, 0.05, timestamp=t)
            states.append(state)
        assert 'sleep' in states, f"final state: {states[-1]}"

    def test_unresponsive_detected(self):
        ms = MicrosleepDetector(ear_close_threshold=0.10)
        t = time.monotonic()
        states = []
        for i in range(38):
            t += 0.04
            state = ms.update(0.05, 0.05, timestamp=t)
            states.append(state)
        for i in range(38):
            t += 0.04
            state = ms.update(0.05, 0.05, timestamp=t)
            states.append(state)
        for i in range(76):
            t += 0.04
            state = ms.update(0.05, 0.05, timestamp=t)
            states.append(state)
        assert 'unresponsive' in states, f"final: {states[-1]}"

    def test_eyes_open_state(self):
        ms = MicrosleepDetector(ear_close_threshold=0.10)
        ms.reset()
        t = time.monotonic()
        t += 0.04
        ms.update(0.30, 0.30, timestamp=t)
        assert ms.state == 'open'


class TestDistraction:
    def test_addw_high_speed_nominal(self):
        dd = DistractionDetector(saccade_tolerance_ms=50, vats_window_sec=30,
                                 vats_threshold_sec=10.0, fps=25)
        dd.set_start_time(0)
        t = time.monotonic()
        for i in range(90):
            t += 0.04
            res = dd.update(is_on_road=False, vehicle_speed=70, dt=0.04, timestamp=t)
        assert res['addw_nominal'], f"timer={res['continuous_offroad']:.2f}s"

    def test_saccade_tolerance(self):
        dd = DistractionDetector(saccade_tolerance_ms=50, vats_window_sec=30,
                                 vats_threshold_sec=10.0, fps=25)
        dd.set_start_time(0)
        t = time.monotonic()
        for i in range(75):
            t += 0.04
            dd.update(False, 70, 0.04, timestamp=t)
        t += 0.04
        res = dd.update(True, 70, 0.04, timestamp=t)
        timer_before = res['continuous_offroad']
        t += 0.04
        res = dd.update(False, 70, 0.04, timestamp=t)
        assert res['continuous_offroad'] > 3.0, \
            f"timer={res['continuous_offroad']:.2f}s (before={timer_before:.2f}s)"

    def test_addw_low_speed_nominal(self):
        dd = DistractionDetector(saccade_tolerance_ms=50, vats_window_sec=30,
                                 vats_threshold_sec=10.0, fps=25)
        dd.set_start_time(0)
        t = time.monotonic()
        for i in range(150):
            t += 0.04
            res = dd.update(False, 30, 0.04, timestamp=t)
        assert res['addw_nominal'], f"timer={res['continuous_offroad']:.2f}s"

    def test_addw_low_speed_not_high_speed(self):
        dd = DistractionDetector(saccade_tolerance_ms=50, vats_window_sec=30,
                                 vats_threshold_sec=10.0, fps=25)
        dd.set_start_time(0)
        t = time.monotonic()
        for i in range(150):
            t += 0.04
            res = dd.update(False, 30, 0.04, timestamp=t)
        assert not res['addw_buffer']

    def test_vats_triggered(self):
        dd = DistractionDetector(saccade_tolerance_ms=50, vats_window_sec=30,
                                 vats_threshold_sec=10.0, fps=25)
        dd.set_start_time(0)
        t = time.monotonic()
        for cycle in range(20):
            for i in range(13):
                t += 0.04
                dd.update(False, 70, 0.04, timestamp=t)
            for i in range(13):
                t += 0.04
                dd.update(True, 70, 0.04, timestamp=t)
        res = dd.update(True, 70, 0.04, timestamp=t)
        assert res['vats_triggered'], f"cumul={res['vats_cumulative']:.1f}s"

    def test_vats_not_triggered_at_low_speed(self):
        dd_low = DistractionDetector(saccade_tolerance_ms=50, vats_window_sec=30,
                                     vats_threshold_sec=10.0, fps=25)
        dd_low.set_start_time(0)
        t = time.monotonic()
        for cycle in range(20):
            for i in range(13):
                t += 0.04
                dd_low.update(False, 15, 0.04, timestamp=t)
            for i in range(13):
                t += 0.04
                dd_low.update(True, 15, 0.04, timestamp=t)
        res = dd_low.update(True, 15, 0.04, timestamp=t)
        assert not res['vats_triggered'], \
            f"cumul={res['vats_cumulative']:.1f}s triggered={res['vats_triggered']}"

    def test_saccade_prescribed_4s_30ms_1s(self):
        dd_sac = DistractionDetector(saccade_tolerance_ms=50, vats_window_sec=30,
                                     vats_threshold_sec=10.0, fps=25)
        dd_sac.set_start_time(0)
        t = time.monotonic()
        for i in range(100):
            t += 0.04
            dd_sac.update(False, 70, 0.04, timestamp=t)
        timer_4s = dd_sac.offroad_timer
        for i in range(1):
            t += 0.04
            dd_sac.update(True, 70, 0.04, timestamp=t)
        for i in range(25):
            t += 0.04
            res = dd_sac.update(False, 70, 0.04, timestamp=t)
        assert abs(res['continuous_offroad'] - 5.03) < 0.2, \
            f"timer={res['continuous_offroad']:.2f}s (4s was={timer_4s:.2f}s)"


class TestDrowsiness:
    def test_normal_kss_low(self):
        ds = DrowsinessScorer(learning_period_sec=600)
        t = time.monotonic()
        res = ds.update('none', 15, False, 'open', 0, 0, timestamp=t)
        assert res['kss'] <= 3, f"KSS={res['kss']}"

    def test_in_learning_period(self):
        ds = DrowsinessScorer(learning_period_sec=600)
        t = time.monotonic()
        res = ds.update('none', 15, False, 'open', 0, 0, timestamp=t)
        assert res['in_learning_period']

    def test_severe_perclos_kss_high(self):
        ds = DrowsinessScorer(learning_period_sec=600)
        t = time.monotonic()
        for i in range(5):
            res = ds.update('severe', 15, False, 'open', 0, 0, timestamp=t)
        assert res['kss'] >= 7, f"KSS={res['kss']}"

    def test_recovery_to_kss1(self):
        ds = DrowsinessScorer(learning_period_sec=600)
        t = time.monotonic()
        ds.reset()
        for i in range(50):
            t += 0.04
            res = ds.update('none', 10, False, 'open', 0, 0, timestamp=t)
        assert res['kss'] == 1, f"KSS={res['kss']} health={res['health_score']:.1f}"

    def test_microsleep_penalty(self):
        ds = DrowsinessScorer(learning_period_sec=600)
        t = time.monotonic()
        ds.reset()
        res = ds.update('none', 10, False, 'microsleep', 0, 0, timestamp=t)
        assert res['health_score'] < 100, f"health={res['health_score']:.1f}"

    def test_head_drop_penalty(self):
        ds = DrowsinessScorer(learning_period_sec=600)
        t = time.monotonic()
        ds.reset()
        res = ds.update('none', 10, False, 'open', 0, 30, timestamp=t)
        assert res['health_score'] < 100, f"health={res['health_score']:.1f}"


class TestWarningManager:
    def test_no_warning_when_normal(self):
        wm = WarningManager(advisory_sec=3.0, escalating_sec=5.0, intervention_sec=5.0)
        res = wm.update(1, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                        'open', vehicle_speed=70)
        assert res['level'] == 'none'

    def test_kss7_escalating_after_3s(self):
        wm = WarningManager(advisory_sec=3.0, escalating_sec=5.0, intervention_sec=5.0)
        t = time.monotonic()
        for i in range(80):
            t += 0.04
            res = wm.update(7, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                            'open', vehicle_speed=70, timestamp=t)
        assert res['level'] == 'escalating', f"level={res['level']}"

    def test_sleep_emergency(self):
        wm = WarningManager(advisory_sec=3.0, escalating_sec=5.0, intervention_sec=5.0)
        t = time.monotonic()
        for i in range(80):
            t += 0.04
            wm.update(7, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                      'open', vehicle_speed=70, timestamp=t)
        t += 0.04
        res = wm.update(8, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                        'sleep', vehicle_speed=70, timestamp=t)
        assert res['level'] == 'emergency', f"level={res['level']}"

    def test_reset_to_none(self):
        wm = WarningManager(advisory_sec=3.0, escalating_sec=5.0, intervention_sec=5.0)
        wm.reset()
        res = wm.update(1, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                        'open', vehicle_speed=70)
        assert res['level'] == 'none'

    def test_distraction_advisory_or_escalating(self):
        wm = WarningManager(advisory_sec=3.0, escalating_sec=5.0, intervention_sec=5.0)
        wm.reset()
        t = time.monotonic()
        for i in range(80):
            t += 0.04
            res = wm.update(1, {'is_distracted': True, 'addw_nominal': True, 'addw_buffer': False},
                            'open', vehicle_speed=70, timestamp=t)
        assert res['level'] in ('advisory', 'escalating'), f"level={res['level']}"

    def test_distraction_buffer_intervention(self):
        wm = WarningManager(advisory_sec=3.0, escalating_sec=5.0, intervention_sec=5.0)
        wm.reset()
        t = time.monotonic()
        for i in range(200):
            t += 0.04
            res = wm.update(1, {'is_distracted': True, 'addw_nominal': True, 'addw_buffer': True},
                            'open', vehicle_speed=70, timestamp=t)
        assert res['level'] in ('intervention', 'emergency'), f"level={res['level']}"

    def test_ddaw_below_70_no_warning(self):
        wm_ddaw = WarningManager(advisory_sec=3.0, escalating_sec=5.0, intervention_sec=5.0)
        t = time.monotonic()
        for i in range(80):
            t += 0.04
            res = wm_ddaw.update(8, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                                 'open', vehicle_speed=65, timestamp=t)
        assert res['level'] == 'none', f"level={res['level']}"

    def test_ddaw_at_70_emergency(self):
        wm_ddaw = WarningManager(advisory_sec=3.0, escalating_sec=5.0, intervention_sec=5.0)
        wm_ddaw.reset()
        t = time.monotonic()
        for i in range(80):
            t += 0.04
            res = wm_ddaw.update(8, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                                 'open', vehicle_speed=70, timestamp=t)
        assert res['level'] == 'emergency', f"level={res['level']}"

    def test_late_emergency_flag(self):
        wm_late = WarningManager(advisory_sec=3.0, escalating_sec=5.0, intervention_sec=5.0)
        t_late = time.monotonic()
        for i in range(80):
            t_late += 0.04
            wm_late.update(7, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                           'open', vehicle_speed=70, timestamp=t_late)
        assert wm_late.level == 'escalating', f"level={wm_late.level}"
        for i in range(140):
            t_late += 0.04
            wm_late.update(7, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                           'open', vehicle_speed=70, timestamp=t_late)
        t_late += 0.04
        wm_late.update(8, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                       'sleep', vehicle_speed=70, timestamp=t_late)
        assert wm_late.late_emergency, \
            f"late_emergency={wm_late.late_emergency}"
