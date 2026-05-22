"""Iris detection from eye region crops.

Supports GoPoint DMS iris_landmark_ptq.tflite model:
  Input: 71x71 RGB float32, normalized [-1, 1]
  Output 0: eye contour landmarks (N, 3)
  Output 1: iris landmarks (N, 3)

Returns iris center (cx, cy) and radius per eye.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import cv2
    HAS_CV2: bool = True
except ImportError:
    HAS_CV2 = False

from .utils.npu import NPUModel


class IrisDetector:
    """Iris center detection from eye region crops.

    Uses GoPoint iris_landmark_ptq.tflite (MediaPipe Iris based).
    Output: iris center (cx, cy) and radius per eye.
    """

    def __init__(self, model_path: str | None = None,
                 input_size: tuple[int, int] = (64, 64),
                 use_npu: bool = True,
                 mock: bool = False) -> None:
        """Initialize the IrisDetector with an iris-landmark TFLite model.

        Args:
            model_path: Path to ``iris_landmark_ptq.tflite``. ``None`` in
                mock mode.
            input_size: ``(width, height)`` expected by the model.
            use_npu: Prefer NPU (Ethos-U65) inference; fall back to CPU.
            mock: Use mock mode returning iris at eye center.
        """
        self.input_size: tuple[int, int] = input_size
        self.mock: bool = mock
        self.model: Any = None

        if not mock and model_path:
            cpu_path = model_path.replace('_vela.tflite', '.tflite')
            self.model = NPUModel(model_path, cpu_fallback_path=cpu_path, use_npu=use_npu)
            if self.model.input_shape is None:
                self.model.set_input_shape(np.array([1, input_size[1], input_size[0], 3]))

    def detect(self, frame: NDArray[np.uint8],
               left_eye_lm: list[tuple[float, float]],
               right_eye_lm: list[tuple[float, float]]) -> tuple[tuple[float, float, float] | None,
                                                                   tuple[float, float, float] | None]:
        """Detect iris centers for both eyes.

        Args:
            frame: RGB numpy array [H, W, 3]
            left_eye_lm: 6 left eye landmarks [(x,y), ...]
            right_eye_lm: 6 right eye landmarks [(x,y), ...]

        Returns:
            (left_iris, right_iris): each (cx, cy, r) in frame coords, or None
        """
        if self.mock:
            return self._mock_detect(left_eye_lm, right_eye_lm)

        left_iris = self._detect_single_eye(frame, left_eye_lm)
        right_iris = self._detect_single_eye(frame, right_eye_lm)
        return left_iris, right_iris

    def _detect_single_eye(self, frame: NDArray[np.uint8],
                           eye_landmarks: list[tuple[float, float]]) -> tuple[float, float, float] | None:
        """Detect iris center and radius in a single eye region.

        Args:
            frame: Full RGB frame ``[H, W, 3]``.
            eye_landmarks: 6 eye landmark points ``[(x, y), ...]``.

        Returns:
            ``(cx, cy, r)`` in frame coordinates, or ``None`` if the eye
            region is too small or the model returns no output.
        """
        pts = np.array(eye_landmarks)
        xmin, ymin = pts.min(axis=0).astype(int)
        xmax, ymax = pts.max(axis=0).astype(int)

        # Add padding (70% of eye size)
        ew, eh = xmax - xmin, ymax - ymin
        pad_x, pad_y = int(ew * 0.7), int(eh * 0.7)
        h, w = frame.shape[:2]
        xmin = max(0, xmin - pad_x)
        ymin = max(0, ymin - pad_y)
        xmax = min(w, xmax + pad_x)
        ymax = min(h, ymax + pad_y)

        roi_w = xmax - xmin
        roi_h = ymax - ymin
        if roi_w < 4 or roi_h < 4:
            return None

        # Crop and preprocess
        eye_crop = frame[ymin:ymax, xmin:xmax]
        ih, iw = self.input_size[1], self.input_size[0]

        if HAS_CV2:
            resized = cv2.resize(eye_crop, (iw, ih))
            if resized.shape[2] == 4:
                resized = resized[:, :, :3]
            input_data = resized.astype(np.float32)
        else:
            input_data = np.zeros((ih, iw, 3), dtype=np.float32)
        input_data = (input_data - 128.0) / 128.0
        input_data = np.expand_dims(input_data, axis=0)

        # Run inference
        outputs = self.model.predict(input_data)
        if not outputs or len(outputs) < 2:
            return None

        # Output 0: eye contour, Output 1: iris landmarks
        raw_iris = outputs[1]
        if raw_iris.ndim == 3:
            raw_iris = raw_iris[0]
        iris_pts = np.reshape(raw_iris, (-1, 3)).astype(np.float32)

        if len(iris_pts) < 1:
            return None

        # Iris center and radius from output points
        # MediaPipe iris outputs: center + 3 contour points
        cx = (iris_pts[0][0] / iw) * roi_w + xmin
        cy = (iris_pts[0][1] / ih) * roi_h + ymin

        if len(iris_pts) >= 4:
            # Compute radius from contour points
            dists: list[float] = []
            for i in range(1, min(4, len(iris_pts))):
                px = (iris_pts[i][0] / iw) * roi_w + xmin
                py = (iris_pts[i][1] / ih) * roi_h + ymin
                dists.append(float(np.sqrt((px - cx)**2 + (py - cy)**2)))
            r = float(np.mean(dists)) if dists else 2.0
        else:
            r = roi_w * 0.15

        return (float(cx), float(cy), float(max(r, 1.0)))

    def _mock_detect(self, left_eye_lm: list[tuple[float, float]],
                     right_eye_lm: list[tuple[float, float]]) -> tuple[tuple[float, float, float],
                                                                        tuple[float, float, float]]:
        """Return mock iris at eye center with estimated radius.

        Args:
            left_eye_lm: 6 left eye landmark points.
            right_eye_lm: 6 right eye landmark points.

        Returns:
            ``(left_iris, right_iris)`` each as ``(cx, cy, r)``.
        """
        def eye_center(eye_lm: list[tuple[float, float]]) -> tuple[float, float, float]:
            xs = [p[0] for p in eye_lm]
            ys = [p[1] for p in eye_lm]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            r = (max(xs) - min(xs)) * 0.15
            return (cx, cy, max(r, 2.0))

        return eye_center(left_eye_lm), eye_center(right_eye_lm)
