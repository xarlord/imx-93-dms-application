"""YAML configuration loader with validation."""
from typing import Any, ClassVar

import yaml
import os


_DEFAULTS: dict[str, Any] = {
    'camera.device': '/dev/video0',
    'camera.width': 1280,
    'camera.height': 800,
    'camera.format': 'YUYV',
    'camera.target_fps': 25,
    'vehicle.speed_source': 'simulate',
    'vehicle.simulated_speed': 0,
    'vehicle.ddaw_activation_speed': 70,
    'vehicle.addw_activation_speed': 20,
    'gaze.yaw_weight': 32.0,
    'gaze.pitch_weight': 45.0,
    'gaze.ppd_yaw': 5.0,
    'gaze.ppd_pitch': 5.0,
    'gaze.stabilization_alpha_fixation': 0.3,
    'gaze.stabilization_alpha_saccade': 0.8,
    'gaze.saccade_velocity_threshold': 15.0,
    'thresholds.ear_close': 0.10,
    'thresholds.ear_blink': 0.19,
    'thresholds.yawn_mar': 0.5,
    'thresholds.yawn_min_duration': 0.5,
    'thresholds.head_yaw_max': 25.0,
    'thresholds.head_pitch_max': 20.0,
    'thresholds.perclos_window_sec': 60,
    'thresholds.perclos_mild': 0.10,
    'thresholds.perclos_severe': 0.20,
    'timing.addw_high_speed_nominal': 3.5,
    'timing.addw_high_speed_buffer': 5.0,
    'timing.addw_high_speed_fn_deadline': 4.0,
    'timing.addw_low_speed_nominal': 6.0,
    'timing.addw_low_speed_buffer': 7.5,
    'timing.addw_low_speed_fn_deadline': 6.5,
    'timing.saccade_tolerance_ms': 50,
    'timing.vats_window_sec': 30,
    'timing.vats_threshold_sec': 10.0,
    'warnings.escalation_advisory_sec': 3.0,
    'warnings.escalation_escalating_sec': 5.0,
    'warnings.escalation_intervention_sec': 5.0,
    'display.width': 1920,
    'display.height': 1080,
    'display.fullscreen': True,
    'display.show_landmarks': True,
    'display.show_gaze_ray': True,
    'display.show_zones': False,
    'calibration.enabled': True,
    'calibration.points': 9,
    'calibration.duration_per_point': 2.0,
    'calibration.learning_period_sec': 600,
    'logging.level': 'INFO',
}


class DMSConfig:
    """YAML-based configuration manager for the DMS application.

    Loads settings from a YAML file (expects a top-level ``dms`` key),
    flattens nested keys with dot notation, and falls back to built-in
    defaults for any missing values.

    Attributes:
        config_path: Absolute path to the loaded YAML file, or None.
    """

    _defaults: ClassVar[dict[str, Any]] = _DEFAULTS

    def __init__(self, config_path: str | None) -> None:
        """Initialise the configuration from a YAML file.

        Args:
            config_path: Path to a YAML config file. If None or the file
                does not exist, the configuration is initialised with
                built-in defaults only.
        """
        self._config_path: str | None = os.path.abspath(config_path) if config_path else None
        self._raw: dict[str, Any] = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self._raw = data.get('dms', {})
        self._flat: dict[str, Any] = self._flatten(self._raw)

    @property
    def config_path(self) -> str | None:
        """Return the absolute path to the loaded config file, or None."""
        return self._config_path

    @staticmethod
    def _flatten(
        d: dict[str, Any],
        parent_key: str = '',
        sep: str = '.',
    ) -> dict[str, Any]:
        """Flatten a nested dictionary into dot-separated keys.

        Args:
            d: The nested dictionary to flatten.
            parent_key: Prefix for keys during recursion.
            sep: Separator string placed between key levels.

        Returns:
            A flat dictionary where nested keys are joined by *sep*.
        """
        items: dict[str, Any] = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(DMSConfig._flatten(v, new_key, sep))
            else:
                items[new_key] = v
        return items

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value by dot-separated key.

        Args:
            key: Dot-separated configuration key (e.g. ``"camera.width"``).
            default: Value to return when the key is absent. If None the
                built-in default table is consulted.

        Returns:
            The configuration value, *default*, or the built-in default.
        """
        if default is None:
            default = _DEFAULTS.get(key)
        return self._flat.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value (for test/override use).

        Args:
            key: Dot-separated configuration key.
            value: The value to store.
        """
        self._flat[key] = value

    def get_section(self, section: str) -> dict[str, Any]:
        """Return all keys within a dot-prefixed section as a flat dict.

        Args:
            section: Section prefix (e.g. ``"camera"``).

        Returns:
            Dictionary mapping stripped sub-keys to their values.
        """
        result: dict[str, Any] = {}
        prefix = section + '.'
        for k, v in self._flat.items():
            if k.startswith(prefix):
                result[k[len(prefix):]] = v
        return result

    @property
    def camera(self) -> dict[str, Any]:
        """Camera capture settings section."""
        return self.get_section('camera')

    @property
    def vehicle(self) -> dict[str, Any]:
        """Vehicle speed and activation-threshold settings section."""
        return self.get_section('vehicle')

    @property
    def gaze(self) -> dict[str, Any]:
        """Gaze estimation weight and stabilisation settings section."""
        return self.get_section('gaze')

    @property
    def thresholds(self) -> dict[str, Any]:
        """EAR, MAR, PERCLOS, and head-pose threshold settings section."""
        return self.get_section('thresholds')

    @property
    def timing(self) -> dict[str, Any]:
        """ADDW timing and VATS window settings section."""
        return self.get_section('timing')

    @property
    def warnings(self) -> dict[str, Any]:
        """Warning escalation duration settings section."""
        return self.get_section('warnings')

    @property
    def display(self) -> dict[str, Any]:
        """Display resolution and overlay settings section."""
        return self.get_section('display')

    @property
    def calibration(self) -> dict[str, Any]:
        """Driver calibration settings section."""
        return self.get_section('calibration')

    @property
    def models(self) -> dict[str, Any]:
        """ML model path settings section."""
        return self.get_section('models')

    @property
    def logging_config(self) -> dict[str, Any]:
        """Logging configuration settings section."""
        return self.get_section('logging')
