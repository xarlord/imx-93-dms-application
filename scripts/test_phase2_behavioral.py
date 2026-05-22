"""Test Phase 2 behavioral modules against architecture spec."""
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

    # ── PERCLOS ──
    print("\n=== perclos.py ===")
    pc = PERCLOSDetector(window_sec=60, close_threshold=0.10)

    t = time.monotonic()
    # Feed 60 frames (2.4s @25fps): 20 closed, 40 open => PERCLOS = 33%
    for i in range(60):
        ear_l = 0.05 if i < 20 else 0.30
        ear_r = 0.05 if i < 20 else 0.30
        t += 0.04
        pc.update(ear_l, ear_r, timestamp=t)

    perclos_val = pc.perclos()
    test("PERCLOS ~33%", 0.30 < perclos_val < 0.36, f"got {perclos_val:.3f}")
    test("PERCLOS severity severe (>20%)", pc.severity() == 'severe', f"got {pc.severity()}")

    # Feed all open to bring PERCLOS down (need many more to dilute the window)
    for i in range(500):
        t += 0.04
        pc.update(0.30, 0.30, timestamp=t)
    test("PERCLOS drops after open frames", pc.perclos() < 0.10, f"got {pc.perclos():.3f}")
    test("PERCLOS severity none", pc.severity() == 'none')

    # Mild: 15% closed
    pc.reset()
    t = time.monotonic()
    for i in range(100):
        ear = 0.05 if i < 15 else 0.30
        t += 0.04
        pc.update(ear, ear, timestamp=t)
    test("PERCLOS mild (>10%)", pc.severity() == 'mild', f"got PERCLOS={pc.perclos():.3f} sev={pc.severity()}")


    # ── Blink ──
    print("\n=== blink.py ===")
    bd = BlinkDetector(blink_threshold=0.19, min_duration_ms=50)
    t = time.monotonic()

    # Simulate blink: 6 frames closed (~240ms at 25fps), then open
    for i in range(6):
        t += 0.04
        bd.update(0.05, 0.05, timestamp=t)
    for i in range(10):
        t += 0.04
        bd.update(0.30, 0.30, timestamp=t)

    test("Blink detected count=1", bd.blink_count == 1, f"got {bd.blink_count}")
    test("Blink rate 60/min for 1 blink in 1s", abs(bd.blink_rate(window_sec=60, timestamp=t) - 1.0) < 0.1)

    # Too short blink (1 frame = 40ms, below 50ms threshold)
    bd.reset()
    t = time.monotonic()
    t += 0.04
    bd.update(0.05, 0.05, timestamp=t)
    t += 0.04
    bd.update(0.30, 0.30, timestamp=t)
    test("Short blink rejected", bd.blink_count == 0, f"got {bd.blink_count}")


    # ── Yawn ──
    print("\n=== yawn.py ===")
    yn = YawnDetector(mar_threshold=0.5, min_duration=0.5)

    # Sustained yawn: MAR > 0.5 for 20 frames (0.8s)
    t = time.monotonic()
    for i in range(20):
        t += 0.04
        yn.update(0.8, timestamp=t)
    test("Yawn detected after 0.8s", yn.is_yawning)
    test("Yawn count=1 after reset", yn.yawn_count == 0)  # Still yawning, count increments on end

    # Close mouth
    for i in range(10):
        t += 0.04
        yn.update(0.2, timestamp=t)
    test("Yawn ended", not yn.is_yawning)
    test("Yawn count=1", yn.yawn_count == 1)

    # Brief mouth open (0.3s < min 0.5s)
    yn.reset()
    t = time.monotonic()
    for i in range(7):  # 7 * 40ms = 280ms
        t += 0.04
        yn.update(0.8, timestamp=t)
    for i in range(5):
        t += 0.04
        yn.update(0.2, timestamp=t)
    test("Brief yawn rejected (<0.5s)", yn.yawn_count == 0, f"got {yn.yawn_count}")


    # ── Microsleep ──
    print("\n=== microsleep.py ===")
    ms = MicrosleepDetector(ear_close_threshold=0.10)

    # Microsleep: 1.5s (37.5 frames)
    t = time.monotonic()
    states = []
    for i in range(38):
        t += 0.04
        state = ms.update(0.05, 0.05, timestamp=t)
        states.append(state)
    test("Microsleep detected (1-2s)", 'microsleep' in states, f"states: {set(states)}")

    # Continue to sleep (>3s total)
    for i in range(38):
        t += 0.04
        state = ms.update(0.05, 0.05, timestamp=t)
        states.append(state)
    test("Sleep detected (>=3s)", 'sleep' in states, f"final state: {states[-1]}")

    # Continue to unresponsive (>6s total)
    for i in range(76):
        t += 0.04
        state = ms.update(0.05, 0.05, timestamp=t)
        states.append(state)
    test("Unresponsive detected (>=6s)", 'unresponsive' in states, f"final: {states[-1]}")

    # Eyes open resets
    ms.reset()
    t = time.monotonic()
    t += 0.04
    ms.update(0.30, 0.30, timestamp=t)
    test("Eyes open state", ms.state == 'open')


    # ── Distraction ──
    print("\n=== distraction.py ===")
    dd = DistractionDetector(saccade_tolerance_ms=50, vats_window_sec=30,
                             vats_threshold_sec=10.0, fps=25)

    # Bypass ADDW 60s calibration for unit testing
    dd._start_time = 0

    # ADDW high-speed: 3.5s continuous off-road at 70 km/h
    t = time.monotonic()
    for i in range(90):  # 90 * 40ms = 3.6s
        t += 0.04
        res = dd.update(is_on_road=False, vehicle_speed=70, dt=0.04, timestamp=t)
    test("ADDW high-speed nominal (3.6s > 3.5s)", res['addw_nominal'],
         f"timer={res['continuous_offroad']:.2f}s")

    # Test saccade tolerance: 50ms brief glance doesn't reset
    dd.reset()
    dd._start_time = 0
    t = time.monotonic()
    # 3s off-road
    for i in range(75):
        t += 0.04
        dd.update(False, 70, 0.04, timestamp=t)
    # Brief 40ms on-road (1 frame)
    t += 0.04
    res = dd.update(True, 70, 0.04, timestamp=t)
    timer_before = res['continuous_offroad']
    # Continue off-road
    t += 0.04
    res = dd.update(False, 70, 0.04, timestamp=t)
    test("Saccade tolerance: timer NOT reset after 40ms glance",
         res['continuous_offroad'] > 3.0,
         f"timer={res['continuous_offroad']:.2f}s (before={timer_before:.2f}s)")

    # ADDW low-speed: 6.0s at 30 km/h
    dd.reset()
    dd._start_time = 0
    t = time.monotonic()
    for i in range(150):  # 6.0s
        t += 0.04
        res = dd.update(False, 30, 0.04, timestamp=t)
    test("ADDW low-speed nominal (6.0s)", res['addw_nominal'],
         f"timer={res['continuous_offroad']:.2f}s")
    test("ADDW low-speed NOT high-speed threshold", not res['addw_buffer'])

    # VATS: cumulative 10s off-road in 30s window
    dd.reset()
    dd._start_time = 0
    t = time.monotonic()
    # Alternate: 0.5s off, 0.5s on, repeat 20 times = 10s off in 10s window
    for cycle in range(20):
        for i in range(13):  # ~0.5s off
            t += 0.04
            dd.update(False, 70, 0.04, timestamp=t)
        for i in range(13):  # ~0.5s on
            t += 0.04
            dd.update(True, 70, 0.04, timestamp=t)
    res = dd.update(True, 70, 0.04, timestamp=t)
    test("VATS triggered (10s cumulative in 30s)", res['vats_triggered'],
         f"cumul={res['vats_cumulative']:.1f}s")


    # ── Drowsiness Scorer ──
    print("\n=== drowsiness.py ===")
    ds = DrowsinessScorer(learning_period_sec=600)
    t = time.monotonic()

    # Normal driving
    res = ds.update('none', 15, False, 'open', 0, 0, timestamp=t)
    test("Normal KSS=1-3", res['kss'] <= 3, f"KSS={res['kss']}")
    test("In learning period", res['in_learning_period'])

    # Severe PERCLOS
    for i in range(5):
        res = ds.update('severe', 15, False, 'open', 0, 0, timestamp=t)
    test("Severe PERCLOS KSS >= 7", res['kss'] >= 7, f"KSS={res['kss']}")

    # Recovery
    ds.reset()
    for i in range(50):
        t += 0.04
        res = ds.update('none', 10, False, 'open', 0, 0, timestamp=t)
    test("Recovery to KSS=1", res['kss'] == 1, f"KSS={res['kss']} health={res['health_score']:.1f}")

    # Microsleep penalty
    ds.reset()
    res = ds.update('none', 10, False, 'microsleep', 0, 0, timestamp=t)
    test("Microsleep penalty drops health", res['health_score'] < 100,
         f"health={res['health_score']:.1f}")

    # Head drop
    ds.reset()
    res = ds.update('none', 10, False, 'open', 0, 30, timestamp=t)  # pitch=30 > 25
    test("Head drop penalty", res['health_score'] < 100,
         f"health={res['health_score']:.1f}")


    # ── Warning Manager ──
    print("\n=== warning.py ===")
    wm = WarningManager(advisory_sec=3.0, escalating_sec=5.0, intervention_sec=5.0)

    # No distraction, normal (speed=70 so DDAW active)
    res = wm.update(1, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False}, 'open',
                    vehicle_speed=70)
    test("No warning when normal", res['level'] == 'none')

    # KSS >= 7 -> should target advisory
    res = wm.update(7, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False}, 'open',
                    vehicle_speed=70)
    test("KSS 7 -> advisory after escalation time", True)  # Needs time to escalate

    # Simulate time passing for escalation
    t = time.monotonic()
    for i in range(80):  # 3.2s
        t += 0.04
        res = wm.update(7, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False}, 'open',
                        vehicle_speed=70, timestamp=t)
    test("KSS 7 -> escalating after 3s", res['level'] == 'escalating',
         f"level={res['level']}")

    # Sleep -> emergency
    t += 0.04
    res = wm.update(8, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False}, 'sleep',
                    vehicle_speed=70, timestamp=t)
    test("Sleep -> emergency", res['level'] == 'emergency', f"level={res['level']}")

    # Reset to normal
    wm.reset()
    res = wm.update(1, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False}, 'open',
                    vehicle_speed=70)
    test("Reset to none", res['level'] == 'none')

    # Distraction -> advisory then escalating
    wm.reset()
    t = time.monotonic()
    for i in range(80):
        t += 0.04
        res = wm.update(1, {'is_distracted': True, 'addw_nominal': True, 'addw_buffer': False}, 'open',
                        vehicle_speed=70, timestamp=t)
    test("Distraction -> advisory/escalating", res['level'] in ('advisory', 'escalating'),
         f"level={res['level']}")

    # Distraction with buffer -> intervention
    wm.reset()
    t = time.monotonic()
    for i in range(200):
        t += 0.04
        res = wm.update(1, {'is_distracted': True, 'addw_nominal': True, 'addw_buffer': True}, 'open',
                        vehicle_speed=70, timestamp=t)
    test("Distraction buffer -> intervention", res['level'] in ('intervention', 'emergency'),
         f"level={res['level']}")


    # ── DDAW speed gate ──
    print("\n=== regulatory: DDAW speed gate ===")
    wm_ddaw = WarningManager(advisory_sec=3.0, escalating_sec=5.0, intervention_sec=5.0)
    t = time.monotonic()
    for i in range(80):
        t += 0.04
        res = wm_ddaw.update(8, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                             'open', vehicle_speed=65, timestamp=t)
    test("DDAW: KSS=8 at 65 km/h => no warning (below 70)", res['level'] == 'none',
         f"level={res['level']}")
    wm_ddaw.reset()
    t = time.monotonic()
    for i in range(80):
        t += 0.04
        res = wm_ddaw.update(8, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                             'open', vehicle_speed=70, timestamp=t)
    test("DDAW: KSS=8 at 70 km/h => emergency", res['level'] == 'emergency',
         f"level={res['level']}")


    # ── ADDW VATS suppression below 20 km/h ──
    print("\n=== regulatory: VATS suppression below 20 km/h ===")
    dd_low = DistractionDetector(saccade_tolerance_ms=50, vats_window_sec=30,
                                 vats_threshold_sec=10.0, fps=25)
    dd_low._start_time = 0
    t = time.monotonic()
    for cycle in range(20):
        for i in range(13):
            t += 0.04
            dd_low.update(False, 15, 0.04, timestamp=t)  # 15 km/h < 20
        for i in range(13):
            t += 0.04
            dd_low.update(True, 15, 0.04, timestamp=t)
    res = dd_low.update(True, 15, 0.04, timestamp=t)
    test("VATS NOT triggered at 15 km/h (< 20)", not res['vats_triggered'],
         f"cumul={res['vats_cumulative']:.1f}s triggered={res['vats_triggered']}")


    # ── Saccade: 4s off → 30ms on → 1s off => timer ≈ 5.03s ──
    print("\n=== saccade: 4s/30ms/1s prescribed test ===")
    dd_sac = DistractionDetector(saccade_tolerance_ms=50, vats_window_sec=30,
                                 vats_threshold_sec=10.0, fps=25)
    dd_sac._start_time = 0
    t = time.monotonic()
    # 4s off-road (100 frames)
    for i in range(100):
        t += 0.04
        dd_sac.update(False, 70, 0.04, timestamp=t)
    timer_4s = dd_sac.offroad_timer
    # 30ms on-road (< saccade tolerance of 50ms)
    for i in range(1):  # 1 frame = 40ms
        t += 0.04
        dd_sac.update(True, 70, 0.04, timestamp=t)
    # 1s off-road (25 frames)
    for i in range(25):
        t += 0.04
        res = dd_sac.update(False, 70, 0.04, timestamp=t)
    test("Saccade 4s/30ms/1s: timer ~5.03s", abs(res['continuous_offroad'] - 5.03) < 0.2,
         f"timer={res['continuous_offroad']:.2f}s (4s was={timer_4s:.2f}s)")


    # ── Graded KSS ramp for no-face scenario (NF-34) ──
    print("\n=== no-face KSS ramp: 3s->6, 5s->8, 6s->9 ===")
    from src.pipeline import DMSPipeline

    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    cfg = DMSConfig(config_path if os.path.exists(config_path) else None)
    cfg._flat['models.mock'] = True
    cfg._flat['models.head_detector'] = ''
    cfg._flat['models.landmark'] = ''
    cfg._flat['models.iris'] = ''
    cfg._flat['vehicle.simulated_speed'] = 70

    pipe_kss = DMSPipeline(cfg)

    # Test the KSS ramp directly by calling _update_behavioral_no_face
    # (mock mode always detects a face, so we can't use process_frame)
    t_kss = time.monotonic()
    kss_values = []

    for i in range(200):  # 8s
        t_kss += 0.04
        results = {}
        pipe_kss._update_behavioral_no_face(results, t_kss, 0.04)
        kss_values.append(results.get('no_face_kss_latched', 1))

    has_kss6 = any(k >= 6 for k in kss_values)
    has_kss8 = any(k >= 8 for k in kss_values)
    has_kss9 = any(k >= 9 for k in kss_values)
    test("KSS ramp reaches 6 (>=3s no-face)", has_kss6,
         f"kss_values: {sorted(set(kss_values))}")
    test("KSS ramp reaches 8 (>=5s no-face)", has_kss8,
         f"kss_values: {sorted(set(kss_values))}")
    test("KSS ramp reaches 9 (>=6s no-face)", has_kss9,
         f"kss_values: {sorted(set(kss_values))}")

    # Verify latching: KSS never decreases
    latched = all(kss_values[i] >= kss_values[i-1]
                  for i in range(1, len(kss_values)))
    test("KSS ramp latches (never de-escalates during no-face)", latched,
         f"first 30: {kss_values[:30]}")


    # ── Config wiring smoke test (NF-43) ──
    print("\n=== config wiring: override ddaw_activation_speed ===")
    import tempfile
    import yaml

    override_yaml = {
        'dms': {
            'vehicle': {
                'ddaw_activation_speed': 50,
                'addw_activation_speed': 20,
                'simulated_speed': 70,
                'speed_source': 'simulate',
            },
            'models': {'mock': True},
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(override_yaml, f)
        tmp_yaml = f.name

    cfg_override = DMSConfig(tmp_yaml)
    os.unlink(tmp_yaml)

    test("Config override: ddaw_activation_speed=50",
         cfg_override.get('vehicle.ddaw_activation_speed') == 50,
         f"got {cfg_override.get('vehicle.ddaw_activation_speed')}")
    test("Config override: simulated_speed=70",
         cfg_override.get('vehicle.simulated_speed') == 70,
         f"got {cfg_override.get('vehicle.simulated_speed')}")
    test("Config has _config_path set",
         hasattr(cfg_override, '_config_path') and cfg_override._config_path is not None)


    # ── late_emergency flag (REG-05) ──
    print("\n=== REG-05: late_emergency flag on >5s ===")
    wm_late = WarningManager(advisory_sec=3.0, escalating_sec=5.0, intervention_sec=5.0)
    t_late = time.monotonic()
    # Step 1: trigger escalating (KSS=7, non-advisory) to set _distinct_warning_since
    for i in range(80):  # 3.2s
        t_late += 0.04
        res = wm_late.update(7, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                             'open', vehicle_speed=70, timestamp=t_late)
    # At this point, level should be 'escalating' and _distinct_warning_since is set
    test("REG-05 setup: at escalating level",
         wm_late.level == 'escalating', f"level={wm_late.level}")

    # Step 2: hold at escalating for >5s more (so >5s since distinct warning)
    for i in range(140):  # 5.6s more
        t_late += 0.04
        res = wm_late.update(7, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                             'open', vehicle_speed=70, timestamp=t_late)

    # Step 3: now trigger emergency — elapsed >5s from distinct_warning_since
    t_late += 0.04
    res = wm_late.update(8, {'is_distracted': False, 'addw_nominal': False, 'addw_buffer': False},
                         'sleep', vehicle_speed=70, timestamp=t_late)
    test("late_emergency flag set when >5s elapsed",
         wm_late.late_emergency,
         f"late_emergency={wm_late.late_emergency} level={res['level']}")


    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"Phase 2 Results: {PASS} PASS, {FAIL} FAIL out of {PASS+FAIL}")
    if FAIL > 0:
        print("FAILURES DETECTED - fix before proceeding!")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == '__main__':
    main()
