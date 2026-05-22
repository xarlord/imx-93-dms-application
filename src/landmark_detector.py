"""Face landmark detection: MediaPipe face landmark model.

Supports GoPoint DMS face_landmark_ptq.tflite model:
  Input: 192x192 RGB float32, normalized [-1, 1]
  Output: (N, 3) landmarks (x, y, z) in model input coordinates

Extracts 25 of 478 MediaPipe landmarks mapped via MEDIAPIPE_TO_DMS.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import cv2
    HAS_CV2: bool = True
except ImportError:
    HAS_CV2 = False

from .utils.npu import NPUModel


# MediaPipe face_landmarker landmark indices mapped to our 25-point scheme
MEDIAPIPE_TO_DMS: dict[int, int] = {
    0: 1,     # Nose tip
    1: 61,    # Left eyebrow inner
    2: 293,   # Right eyebrow inner
    3: 46,    # Left eyebrow outer
    4: 276,   # Right eyebrow outer
    # Left eye: outer(33), upper(160), upper(158), inner(133), lower(153), lower(144)
    5: 33, 6: 160, 7: 158, 8: 133, 9: 153, 10: 144,
    # Right eye: outer(263), upper(387), upper(385), inner(362), lower(380), lower(373)
    11: 263, 12: 387, 13: 385, 14: 362, 15: 380, 16: 373,
    # Mouth: left(61), upper-lip(39), upper-top(11), upper-r(0), right(291),
    #         lower-r(275), lower-bot(281), lower-l(57)
    17: 61, 18: 39, 19: 11, 20: 0, 21: 291, 22: 275, 23: 281, 24: 57,
}

# Reverse mapping: MediaPipe index -> DMS index
MP_TO_DMS: dict[int, int] = {v: k for k, v in MEDIAPIPE_TO_DMS.items()}


class LandmarkDetector:
    """Face landmark detection from cropped head. Input: 192x192 RGB.

    Uses GoPoint face_landmark_ptq.tflite (MediaPipe-based).
    Extracts only 25 of 478 output landmarks.
    """

    def __init__(self, model_path: str | None = None,
                 input_size: tuple[int, int] = (192, 192),
                 use_npu: bool = True,
                 mock: bool = False) -> None:
        """Initialize the LandmarkDetector with a face-landmark TFLite model.

        Args:
            model_path: Path to ``face_landmark_ptq.tflite``. ``None`` in
                mock mode.
            input_size: ``(width, height)`` expected by the model.
            use_npu: Prefer NPU (Ethos-U65) inference; fall back to CPU.
            mock: Use mock mode returning synthetic landmarks.
        """
        self.input_size: tuple[int, int] = input_size
        self.mock: bool = mock
        self.model: object | None = None

        if not mock and model_path:
            cpu_path = model_path.replace('_vela.tflite', '.tflite')
            self.model = NPUModel(model_path, cpu_fallback_path=cpu_path, use_npu=use_npu)
            if self.model.input_shape is None:
                self.model.set_input_shape(np.array([1, input_size[1], input_size[0], 3]))

    def detect(self, head_crop: NDArray[np.uint8],
               head_bbox: list[float]) -> list[tuple[float, float]] | None:
        """Detect 25 landmarks in a cropped head image.

        Args:
            head_crop: RGB numpy array of cropped head
            head_bbox: [x1, y1, x2, y2] in original frame coords

        Returns:
            list of 25 (x, y) tuples in original frame coordinates,
            or None if no face detected.
        """
        if self.mock:
            return self._mock_detect(head_crop, head_bbox)

        ih, iw = self.input_size[1], self.input_size[0]

        # Preprocess: resize to 192x192, normalize [-1, 1]
        if HAS_CV2:
            resized = cv2.resize(head_crop, (iw, ih))
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

        raw = outputs[-1]
        if raw.ndim == 3:
            raw = raw[0]
        raw = raw.astype(np.float32)
        if raw.size < 3:
            return None
        raw = np.reshape(raw, (-1, 3))

        # Extract 25 DMS landmarks from MediaPipe landmarks
        landmarks: list[tuple[float, float]] = []
        for dms_idx in range(25):
            mp_idx = MEDIAPIPE_TO_DMS[dms_idx]
            if mp_idx < raw.shape[0]:
                # Normalize to [0,1] then map to frame coords
                x_norm = raw[mp_idx][0] / iw
                y_norm = raw[mp_idx][1] / ih
                x_frame = x_norm * (head_bbox[2] - head_bbox[0]) + head_bbox[0]
                y_frame = y_norm * (head_bbox[3] - head_bbox[1]) + head_bbox[1]
                landmarks.append((float(x_frame), float(y_frame)))
            else:
                landmarks.append((0.0, 0.0))

        return landmarks

    def _mock_detect(self, head_crop: NDArray[np.uint8],
                     head_bbox: list[float]) -> list[tuple[float, float]]:
        """Generate synthetic 25 landmarks within the head bbox."""
        x1, y1, x2, y2 = head_bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        landmarks: list[tuple[float, float]] = [
            (cx, cy - h * 0.05),
            (cx - w * 0.18, cy - h * 0.20),
            (cx + w * 0.18, cy - h * 0.20),
            (cx - w * 0.30, cy - h * 0.22),
            (cx + w * 0.30, cy - h * 0.22),
            (cx - w * 0.25, cy - h * 0.08),
            (cx - w * 0.18, cy - h * 0.12),
            (cx - w * 0.10, cy - h * 0.12),
            (cx - w * 0.05, cy - h * 0.08),
            (cx - w * 0.10, cy - h * 0.04),
            (cx - w * 0.18, cy - h * 0.04),
            (cx + w * 0.25, cy - h * 0.08),
            (cx + w * 0.18, cy - h * 0.12),
            (cx + w * 0.10, cy - h * 0.12),
            (cx + w * 0.05, cy - h * 0.08),
            (cx + w * 0.10, cy - h * 0.04),
            (cx + w * 0.18, cy - h * 0.04),
            (cx - w * 0.15, cy + h * 0.12),
            (cx - w * 0.08, cy + h * 0.08),
            (cx,           cy + h * 0.07),
            (cx + w * 0.08, cy + h * 0.08),
            (cx + w * 0.15, cy + h * 0.12),
            (cx + w * 0.08, cy + h * 0.16),
            (cx,           cy + h * 0.17),
            (cx - w * 0.08, cy + h * 0.16),
        ]
        return landmarks
