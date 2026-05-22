"""NPU interpreter wrapper with CPU fallback."""
from typing import Any

import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger('dms.npu')


class NPUModel:
    """Unified wrapper for NPU inference with automatic CPU fallback.

    Handles INT8 quantized models correctly: when input dtype is uint8/int8,
    applies quantization parameters (scale, zero_point) before passing data
    to the interpreter. Rejects mismatched input with a clear error.
    """

    def __init__(
        self,
        model_path: str,
        cpu_fallback_path: str | None = None,
        use_npu: bool = True,
    ) -> None:
        """Initialise the model, trying NPU backends before CPU fallback.

        The loading order when *use_npu* is True:
        1. Ethos-U direct interpreter on *model_path*.
        2. Ethos-U direct interpreter on *cpu_fallback_path* (if given).
        3. TFLite with Ethos-U delegate on *model_path*.
        4. TFLite with Ethos-U delegate on *cpu_fallback_path* (if given).
        5. Pure CPU TFLite interpreter as final fallback.

        Args:
            model_path: Path to the primary TFLite model file.
            cpu_fallback_path: Optional path to a non-NPU model variant
                used when NPU loading fails.
            use_npu: If False, skip all NPU attempts and go straight to
                CPU inference.
        """
        self.api: str | None = None
        self.interpreter: Any = None
        self.input_details: list[dict[str, Any]] = []
        self.output_details: list[dict[str, Any]] = []
        self.input_shape: tuple[int, ...] | None = None
        self._input_dtype: type[np.floating[Any] | np.integer[Any]] | type[np.uint8] = np.float32
        self._quant_scale: float | None = None
        self._quant_zero_point: int | None = None

        if use_npu:
            loaded = self._try_load_delegate(model_path)
            if not loaded and cpu_fallback_path:
                loaded = self._try_load_delegate(cpu_fallback_path)
            if not loaded:
                loaded = self._try_load_ethosu(model_path)
            if not loaded and cpu_fallback_path:
                loaded = self._try_load_ethosu(cpu_fallback_path)

        if self.api is None:
            path = cpu_fallback_path or model_path
            self._load_cpu(path)

    def _try_load_ethosu(self, model_path: str) -> bool:
        """Attempt to load the model with the ethosu direct interpreter.

        After loading, opens a shadow CPU interpreter on the same model to
        extract input/output tensor details (shape, dtype, quantization)
        that the ethosu API does not expose.

        Args:
            model_path: Path to the TFLite model file.

        Returns:
            True if loading succeeded, False otherwise.
        """
        try:
            import ethosu.interpreter as ethosu
            self.interpreter = ethosu.Interpreter(model_path)
            self.api = 'ethosu'
            self._extract_details_ethosu()
            self._load_shadow_details(model_path)
            return True
        except (ImportError, OSError, RuntimeError):
            logger.debug('ethosu interpreter unavailable for %s', model_path)
            return False

    def _try_load_delegate(self, model_path: str) -> bool:
        """Attempt to load the model with TFLite plus the Ethos-U delegate.

        Args:
            model_path: Path to the TFLite model file.

        Returns:
            True if loading succeeded, False otherwise.
        """
        try:
            import tflite_runtime.interpreter as tflite
            delegate = tflite.load_delegate("/usr/lib/libethosu_delegate.so")
            self.interpreter = tflite.Interpreter(
                model_path=model_path, experimental_delegates=[delegate])
            self.interpreter.allocate_tensors()
            self.api = 'tflite_delegate'
            self._extract_details_tflite()
            return True
        except (ImportError, OSError, RuntimeError):
            logger.debug('ethosu delegate unavailable for %s', model_path)
            return False

    def _load_cpu(self, model_path: str) -> None:
        """Load the model with a plain CPU TFLite interpreter.

        Args:
            model_path: Path to the TFLite model file.
        """
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.api = 'tflite_cpu'
        self._extract_details_tflite()

    def _extract_details_ethosu(self) -> None:
        """Set minimal input/output details for the ethosu API.

        The ethosu interpreter does not expose tensor introspection, so
        defaults are set and must be overridden via :meth:`set_input_shape`
        and :meth:`set_quantization`.
        """
        self.input_details = []
        self.output_details = []
        self.input_shape = None
        self._input_dtype = np.uint8
        self._quant_scale = None
        self._quant_zero_point = None

    def _load_shadow_details(self, model_path: str) -> None:
        """Load a shadow CPU interpreter to extract tensor details.

        The ethosu interpreter does not expose get_input_details() or
        get_output_details().  Opening a second CPU interpreter on the
        same model file lets us read shape, dtype, and quantization
        parameters so that :meth:`_preprocess_input` works correctly.

        If the model is Vela-compiled and cannot run on CPU alone, this
        silently falls back to the ethosu defaults (caller must use
        :meth:`set_input_shape` and :meth:`set_quantization`).

        Args:
            model_path: Path to the same model file loaded by ethosu.
        """
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                return
        try:
            shadow = tflite.Interpreter(model_path=model_path)
            shadow.allocate_tensors()
            inp = shadow.get_input_details()
            out = shadow.get_output_details()
            if inp:
                detail = inp[0]
                self.input_shape = tuple(int(x) for x in detail['shape'])
                self._input_dtype = detail.get('dtype', np.float32)
                qparams = detail.get('quantization_parameters', None)
                if qparams and len(qparams.get('scales', [])) > 0:
                    self._quant_scale = qparams['scales'][0]
                    self._quant_zero_point = qparams['zero_points'][0]
            if out:
                self.output_details = out
            self.input_details = inp
            logger.debug(
                'shadow CPU details for ethosu: shape=%s dtype=%s quant=%s',
                self.input_shape, self._input_dtype,
                'yes' if self._quant_scale else 'no',
            )
        except Exception:
            logger.debug('shadow CPU interpreter failed for %s', model_path)

    def _extract_details_tflite(self) -> None:
        """Populate input/output details and quantization parameters from
        a TFLite interpreter.
        """
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if self.input_details:
            detail = self.input_details[0]
            self.input_shape = tuple(int(x) for x in detail['shape'])
            self._input_dtype = detail.get('dtype', np.float32)
            qparams = detail.get('quantization_parameters', None)
            if qparams and len(qparams.get('scales', [])) > 0:
                self._quant_scale = qparams['scales'][0]
                self._quant_zero_point = qparams['zero_points'][0]

    def set_input_shape(self, shape: tuple[int, ...]) -> None:
        """Set the expected input shape.

        Needed for the ethosu API which lacks automatic introspection.

        Args:
            shape: The input tensor shape, e.g. ``(1, 192, 192, 3)``.
        """
        self.input_shape = tuple(int(x) for x in shape)

    def set_quantization(
        self,
        scale: float,
        zero_point: int,
        dtype: type = np.uint8,
    ) -> None:
        """Override quantization parameters for the ethosu inference path.

        Args:
            scale: The quantization scale factor.
            zero_point: The quantization zero-point value.
            dtype: The expected integer dtype after quantization.
        """
        self._quant_scale = scale
        self._quant_zero_point = zero_point
        self._input_dtype = dtype

    def _preprocess_input(self, input_data: NDArray[Any]) -> NDArray[Any]:
        """Apply quantization if the model expects integer input but received float.

        Args:
            input_data: Raw input array.

        Returns:
            The array cast (and optionally quantized) to the model's
            expected dtype.

        Raises:
            TypeError: If the model expects integer input, float data was
                provided, and no quantization parameters have been set.
        """
        target_dtype = self._input_dtype

        if np.issubdtype(target_dtype, np.integer):
            if np.issubdtype(input_data.dtype, np.floating):
                if self._quant_scale is not None and self._quant_zero_point is not None:
                    q = np.round(
                        input_data / self._quant_scale + self._quant_zero_point
                    )
                    if np.issubdtype(target_dtype, np.unsignedinteger):
                        q = np.clip(q, 0, np.iinfo(target_dtype).max)
                    else:
                        q = np.clip(q, np.iinfo(target_dtype).min, np.iinfo(target_dtype).max)
                    input_data = q.astype(target_dtype)
                else:
                    raise TypeError(
                        f'Model expects {target_dtype} input but received '
                        f'{input_data.dtype}. No quantization parameters set — '
                        f'call set_quantization(scale, zero_point) or pass '
                        f'pre-quantized input.'
                    )
            else:
                input_data = input_data.astype(target_dtype)
        else:
            input_data = input_data.astype(target_dtype)

        return input_data

    def predict(self, input_data: NDArray[Any]) -> list[NDArray[Any]]:
        """Run inference on the loaded model.

        The input is reshaped to the expected shape when the total number
        of elements matches, and quantized automatically when needed.

        Args:
            input_data: Input tensor as a NumPy array.

        Returns:
            A list of output NumPy arrays from the model.

        Raises:
            ValueError: If the input shape differs from the expected shape
                in total element count.
        """
        if self.input_shape is not None:
            expected = self.input_shape
            actual = input_data.shape
            if actual != expected:
                if actual[0] == expected[0] and np.prod(actual) == np.prod(expected):
                    raise ValueError(
                        f'Input shape {actual} differs from expected {expected}. '
                        f'Total elements match but layout may differ (e.g. NHWC vs NCHW). '
                        f'Reshape the input before calling predict().'
                    )
                input_data = input_data.reshape(expected)

        input_data = self._preprocess_input(input_data)

        if self.api == 'ethosu':
            self.interpreter.set_input(0, input_data)
            self.interpreter.invoke()
            outputs: list[NDArray[Any]] = []
            for i in range(4):
                try:
                    out = self.interpreter.get_output(i)
                    if out is not None:
                        outputs.append(np.array(out))
                except (IndexError, RuntimeError):
                    break
            return outputs if outputs else [np.array([])]
        else:
            idx = self.input_details[0]['index']
            self.interpreter.set_tensor(idx, input_data)
            self.interpreter.invoke()
            return [self.interpreter.get_tensor(d['index'])
                    for d in self.output_details]

    @property
    def has_quantization(self) -> bool:
        """Whether quantization parameters have been set."""
        return self._quant_scale is not None

    @property
    def backend(self) -> str:
        """Name of the active inference backend."""
        return self.api or 'unknown'
