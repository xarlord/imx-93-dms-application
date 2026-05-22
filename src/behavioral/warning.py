"""Warning escalation manager: 5-tier hierarchy per regulatory spec."""
import logging
import os
import queue
import subprocess
import threading
import time
from typing import Any, Callable, ClassVar

logger = logging.getLogger('dms.warning')


class WarningManager:
    """Manages warning level escalation per spec:
    none -> advisory -> escalating -> intervention -> emergency
    Escalation timing per config. Microsleep/sleep must have higher urgency.
    Emergency function must start <= 5s after distinct warning (Euro NCAP).

    Regulatory gating:
      - DDAW (drowsiness) warnings only active >= ddaw_activation_speed (70 km/h)
      - ADDW (distraction) warnings only active >= addw_activation_speed (20 km/h)

    Cumulative tracking policy: warning_active_since and _distinct_warning_since
    are RESET on de-escalation to 'none'. This means each warning episode starts
    fresh. Euro NCAP timing applies within a single episode, not across episodes.
    """

    LEVELS: ClassVar[list[str]] = ['none', 'advisory', 'escalating', 'intervention', 'emergency']

    def __init__(self, advisory_sec: float = 3.0, escalating_sec: float = 5.0,
                 intervention_sec: float = 5.0,
                 ddaw_activation_speed: int = 70, addw_activation_speed: int = 20,
                 acoustic_advisory: str | None = None,
                 acoustic_escalating: str | None = None,
                 acoustic_intervention: str | None = None,
                 acoustic_emergency: str | None = None,
                 haptic_callback: Callable[[str], None] | None = None,
                 config_dir: str | None = None) -> None:
        """Initializes the WarningManager with escalation timing and feedback channels.

        Args:
            advisory_sec: Seconds at advisory level before escalating.
            escalating_sec: Seconds at escalating level before escalating.
            intervention_sec: Seconds at intervention level before escalating.
            ddaw_activation_speed: Minimum vehicle speed in km/h for DDAW warnings.
            addw_activation_speed: Minimum vehicle speed in km/h for ADDW warnings.
            acoustic_advisory: Path to WAV file for advisory acoustic warning.
            acoustic_escalating: Path to WAV file for escalating acoustic warning.
            acoustic_intervention: Path to WAV file for intervention acoustic warning.
            acoustic_emergency: Path to WAV file for emergency acoustic warning.
            haptic_callback: Callable invoked with the level name for haptic feedback.
            config_dir: Base directory used to resolve relative acoustic file paths.
        """
        self.escalation_durations: dict[str, float] = {
            'advisory': advisory_sec,
            'escalating': escalating_sec,
            'intervention': intervention_sec,
        }
        self.ddaw_activation_speed: int = ddaw_activation_speed
        self.addw_activation_speed: int = addw_activation_speed
        self.level: str = 'none'
        self.level_start: float | None = None
        self.warning_active_since: float | None = None
        self._distinct_warning_since: float | None = None
        self.late_emergency: bool = False

        self._haptic_callback: Callable[[str], None] = haptic_callback or self._noop_haptic

        self._config_dir: str | None = config_dir
        self._acoustic_files: dict[str, str | None] = {
            'advisory': self._resolve_path(acoustic_advisory),
            'escalating': self._resolve_path(acoustic_escalating),
            'intervention': self._resolve_path(acoustic_intervention),
            'emergency': self._resolve_path(acoustic_emergency),
        }
        self._last_acoustic_level: str | None = None
        self._last_acoustic_time: float = 0
        self._acoustic_cooldown: float = 2.0

        self._audio_queue: queue.Queue[str | None] = queue.Queue(maxsize=4)
        self._audio_thread: threading.Thread = threading.Thread(target=self._audio_worker, daemon=True)
        self._audio_thread.start()

    def _resolve_path(self, path: str | None) -> str | None:
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        if self._config_dir:
            return os.path.join(self._config_dir, path)
        return path

    @staticmethod
    def _noop_haptic(level: str) -> None:
        logger.debug('Haptic callback (no-op): level=%s', level)

    def _audio_worker(self) -> None:
        while True:
            try:
                wav = self._audio_queue.get(timeout=1.0)
                if wav is None:
                    break
                try:
                    subprocess.run(['aplay', '-q', wav],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=3.0)
                except FileNotFoundError:
                    logger.debug('aplay not available, skipping acoustic warning')
                except subprocess.SubprocessError as e:
                    logger.warning('Acoustic warning failed: %s', e)
            except queue.Empty:
                continue

    def _compute_target_level(self, kss: int | float,
                              distraction_result: dict[str, Any],
                              microsleep_state: str,
                              drowsiness_enabled: bool,
                              distraction_enabled: bool) -> str:
        """Determines the target warning level from behavioral inputs.

        Args:
            kss: Karolinska Sleepiness Scale score (1-9).
            distraction_result: Distraction detection result dict.
            microsleep_state: Current microsleep/sleep state string.
            drowsiness_enabled: Whether DDAW gating speed threshold is met.
            distraction_enabled: Whether ADDW gating speed threshold is met.

        Returns:
            Target warning level name (``'none'``, ``'advisory'``,
            ``'escalating'``, ``'intervention'``, or ``'emergency'``).
        """
        is_distraction = (distraction_result.get('addw_nominal', False) or
                          distraction_result.get('addw_buffer', False) or
                          distraction_result.get('is_distracted', False))
        target_level = 'none'

        if microsleep_state == 'unresponsive' or kss >= 8:
            if drowsiness_enabled:
                target_level = 'emergency'
        elif microsleep_state == 'sleep':
            if drowsiness_enabled:
                target_level = 'emergency'
        elif microsleep_state == 'microsleep' or kss >= 7:
            if drowsiness_enabled:
                target_level = 'escalating'
        elif distraction_result.get('addw_buffer', False):
            if distraction_enabled:
                target_level = 'intervention'
        elif is_distraction:
            if distraction_enabled:
                target_level = 'escalating'
        elif distraction_result.get('addw_nominal', False):
            if distraction_enabled:
                target_level = 'advisory'
        elif kss >= 6:
            if drowsiness_enabled:
                target_level = 'advisory'

        return target_level

    def _check_late_emergency(self, target_level: str, now: float) -> None:
        """Flags ``late_emergency`` if emergency escalation exceeds the 5 s Euro NCAP limit.

        Args:
            target_level: The newly computed target warning level.
            now: Current monotonic timestamp.
        """
        self.late_emergency = False
        if target_level == 'emergency' and self._distinct_warning_since is not None:
            elapsed_since_warning = now - self._distinct_warning_since
            if elapsed_since_warning > 5.0 and self.level != 'emergency':
                logger.error('EMERGENCY ESCALATION %.1fs after distinct warning '
                             '(Euro NCAP limit: 5s). Flagging late_emergency.',
                             elapsed_since_warning)
                self.late_emergency = True

    def _apply_escalation_timing(self, target_level: str, now: float) -> None:
        """Transitions the warning level respecting configured escalation durations.

        Emergency level bypasses the duration gate. De-escalation to ``'none'``
        resets cumulative timing state.

        Args:
            target_level: The desired warning level.
            now: Current monotonic timestamp.
        """
        current_idx = self.LEVELS.index(self.level)
        target_idx = self.LEVELS.index(target_level)

        bypass_timing = target_level == 'emergency'

        if target_idx > current_idx:
            if self.level_start is None:
                self.level_start = now
            if self.warning_active_since is None:
                self.warning_active_since = now
            if self._distinct_warning_since is None and target_level != 'advisory':
                self._distinct_warning_since = now
            if bypass_timing:
                self.level = target_level
                self.level_start = now
            else:
                duration_at_level = now - self.level_start
                max_dur = self.escalation_durations.get(self.level, 0)
                if duration_at_level >= max_dur:
                    self.level = target_level
                    self.level_start = now
        elif target_idx < current_idx:
            if target_level == 'none':
                self.level_start = None
                self.warning_active_since = None
                self._distinct_warning_since = None
            else:
                self.level_start = now
            self.level = target_level

    def update(self, kss: int | float,
               distraction_result: dict[str, Any],
               microsleep_state: str,
               vehicle_speed: float = 0,
               timestamp: float | None = None) -> dict[str, Any]:
        """Evaluates inputs and advances the warning level per escalation timing.

        Args:
            kss: Karolinska Sleepiness Scale score (1-9).
            distraction_result: Distraction detection result dict with keys such as
                ``addw_nominal``, ``addw_buffer``, and ``is_distracted``.
            microsleep_state: Current microsleep/sleep state string
                (``'open'``, ``'closed'``, ``'microsleep'``, ``'sleep'``,
                ``'unresponsive'``).
            vehicle_speed: Current vehicle speed in km/h.
            timestamp: Monotonic timestamp override; defaults to ``time.monotonic()``.

        Returns:
            Dict with keys ``level``, ``level_idx``, ``duration_at_level``,
            ``warning_active_for``, ``drowsiness_enabled``, ``distraction_enabled``,
            and ``late_emergency``.
        """
        now = timestamp or time.monotonic()

        drowsiness_enabled = vehicle_speed >= self.ddaw_activation_speed
        distraction_enabled = vehicle_speed >= self.addw_activation_speed

        target_level = self._compute_target_level(
            kss, distraction_result, microsleep_state,
            drowsiness_enabled, distraction_enabled)
        self._check_late_emergency(target_level, now)
        self._apply_escalation_timing(target_level, now)
        self._dispatch_warnings(now)

        return {
            'level': self.level,
            'level_idx': self.LEVELS.index(self.level),
            'duration_at_level': (now - self.level_start) if self.level_start is not None else 0.0,
            'warning_active_for': (now - self.warning_active_since) if self.warning_active_since is not None else 0.0,
            'drowsiness_enabled': drowsiness_enabled,
            'distraction_enabled': distraction_enabled,
            'late_emergency': self.late_emergency,
        }

    def _dispatch_warnings(self, now: float) -> None:
        """Dispatches acoustic and haptic warnings based on current level.

        Args:
            now: Current monotonic timestamp.
        """
        if self.level == 'none':
            return

        wav = self._acoustic_files.get(self.level)
        if wav and os.path.exists(wav):
            level_idx = self.LEVELS.index(self.level)
            last_idx = (self.LEVELS.index(self._last_acoustic_level)
                        if self._last_acoustic_level else -1)
            if level_idx >= last_idx:
                if (self._last_acoustic_level != self.level or
                        (now - self._last_acoustic_time) >= self._acoustic_cooldown):
                    try:
                        self._audio_queue.put_nowait(wav)
                        self._last_acoustic_level = self.level
                        self._last_acoustic_time = now
                    except queue.Full:
                        pass

        if self.level in ('intervention', 'emergency'):
            try:
                self._haptic_callback(self.level)
            except (RuntimeError, OSError) as e:
                logger.warning('Haptic callback error: %s', e)

    def stop(self) -> None:
        """Stops the audio worker thread gracefully."""
        self._audio_queue.put(None)
        self._audio_thread.join(timeout=2.0)

    @property
    def is_active(self) -> bool:
        """Returns whether a warning level above 'none' is currently active.

        Returns:
            True if the current level is not ``'none'``.
        """
        return self.level != 'none'

    def reset(self) -> None:
        """Resets the warning manager to the initial 'none' state."""
        self.level = 'none'
        self.level_start = None
        self.warning_active_since = None
        self._distinct_warning_since = None
        self.late_emergency = False
