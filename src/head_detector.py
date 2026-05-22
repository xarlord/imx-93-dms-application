"""Face detection stage: MediaPipe BlazeFace-based SSD detector.

Supports GoPoint DMS face_detection_ptq.tflite model:
  Input: 128x128 RGB float32, normalized [-1, 1]
  Output: raw_boxes (896, 16) + scores (896, 1) with SSD anchor decoding
"""
from __future__ import annotations

import operator
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import cv2
    HAS_CV2: bool = True
except ImportError:
    HAS_CV2 = False

from .utils.npu import NPUModel


class HeadDetector:
    """MediaPipe BlazeFace face detector. Input: 128x128 RGB, Output: face bbox.

    Supports NPU model (via NPUModel wrapper) or mock mode for testing.
    """

    def __init__(self, model_path: str | None = None,
                 input_size: tuple[int, int] = (128, 128),
                 confidence_threshold: float = 0.7,
                 iou_threshold: float = 0.5,
                 use_npu: bool = True,
                 mock: bool = False) -> None:
        """Initialize the HeadDetector with a BlazeFace TFLite model.

        Args:
            model_path: Path to the face_detection_ptq.tflite model file.
                If ``None`` and ``mock`` is ``False``, no model is loaded.
            input_size: ``(width, height)`` expected by the model input.
            confidence_threshold: Minimum score to keep a detection.
            iou_threshold: Intersection-over-union threshold for NMS.
            use_npu: Prefer NPU (Ethos-U65) inference; fall back to CPU.
            mock: Use mock mode returning a synthetic centered face bbox.

        Raises:
            ValueError: If the model input shape does not match *input_size*.
        """
        self.input_size: tuple[int, int] = input_size
        self.confidence_threshold: float = confidence_threshold
        self.iou_threshold: float = iou_threshold
        self.mock: bool = mock
        self.model: Any = None
        self.num_boxes: int = 896

        if not mock and model_path:
            cpu_path = model_path.replace('_vela.tflite', '.tflite')
            self.model = NPUModel(model_path, cpu_fallback_path=cpu_path, use_npu=use_npu)
            if self.model.input_shape is None:
                self.model.set_input_shape([1, input_size[1], input_size[0], 3])
            else:
                expected_hw = (input_size[1], input_size[0])
                actual_hw = self.model.input_shape[1:3]
                if actual_hw != expected_hw:
                    raise ValueError(
                        f'Model input shape {self.model.input_shape} does not match '
                        f'expected (1, {expected_hw[0]}, {expected_hw[1]}, 3). '
                        f'Check model_path or input_size.'
                    )
            # If ethosu path and no quant params from introspection, set them
            # BlazeFace PTQ model: float [-1,1] normalized input, quantized to uint8
            if self.model.api == 'ethosu' and not self.model.has_quantization:
                self.model.set_quantization(scale=2.0 / 255.0, zero_point=128,
                                            dtype=np.uint8)
            # Generate SSD anchors for BlazeFace
            self.anchors: NDArray[np.float32] = self._generate_anchors()

    def _preprocess(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Resize and normalize a frame to model input range [-1, 1].

        Args:
            frame: BGR or BGRx image array ``[H, W, 3 or 4]``.

        Returns:
            Float32 array of shape ``[1, H_in, W_in, 3]`` normalized to [-1, 1].
        """
        ih, iw = self.input_size[1], self.input_size[0]
        if HAS_CV2:
            resized = cv2.resize(frame, (iw, ih))
            if resized.shape[2] == 4:
                resized = resized[:, :, :3]
            input_data = resized.astype(np.float32)
        else:
            input_data = np.zeros((ih, iw, 3), dtype=np.float32)
        input_data = (input_data - 128.0) / 128.0
        input_data = np.expand_dims(input_data, axis=0)
        return input_data

    def _run_inference(self, input_data: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
        """Run TFLite inference and apply sigmoid to raw logit scores.

        Args:
            input_data: Preprocessed float32 input tensor ``[1, H, W, 3]``.

        Returns:
            Tuple of ``(raw_boxes, raw_scores)`` arrays, or ``None`` if the
            model returned fewer than two output tensors.
        """
        outputs = self.model.predict(input_data)
        if not outputs or len(outputs) < 2:
            return None

        raw_boxes, raw_scores = self._sort_outputs(outputs)

        if raw_boxes.ndim == 3:
            raw_boxes = raw_boxes[0]
        if raw_scores.ndim == 3:
            raw_scores = raw_scores[0]
        if raw_scores.ndim > 1:
            raw_scores = raw_scores[..., 0]
        raw_scores = np.clip(raw_scores, -80, 80)
        raw_scores = 1.0 / (1.0 + np.exp(-raw_scores))
        return raw_boxes, raw_scores

    def _sort_outputs(self, outputs: list[NDArray[Any]]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Return (boxes, scores) in consistent order regardless of backend.

        The Vela optimizer may reorder model outputs.  This method uses the
        output tensor names from ``output_details`` to identify regressors
        (boxes) vs classificators (scores).  Falls back to shape-based
        detection when details are unavailable.

        Args:
            outputs: Raw output tensors from ``NPUModel.predict()``.

        Returns:
            Tuple of ``(boxes_tensor, scores_tensor)``.
        """
        out_details = self.model.output_details if self.model else []
        if out_details and len(out_details) == len(outputs):
            for detail, tensor in zip(out_details, outputs):
                name: str = detail.get('name', '')
                if 'regressor' in name:
                    boxes = tensor
                elif 'classificator' in name:
                    scores = tensor
            if 'boxes' in dir() and 'scores' in dir():
                return boxes, scores

        if len(outputs) >= 2:
            if outputs[0].shape[-1] > outputs[1].shape[-1]:
                return outputs[0], outputs[1]
            return outputs[1], outputs[0]

        return outputs[0], outputs[1]

    def _decode_and_filter(self, raw_boxes: NDArray[np.float32],
                           raw_scores: NDArray[np.float32],
                           h: int, w: int) -> list[dict[str, Any]]:
        """Decode SSD boxes, filter by confidence, and scale to frame pixels.

        Args:
            raw_boxes: Raw box predictions ``[N, 16]``.
            raw_scores: Confidence scores ``[N]`` after sigmoid.
            h: Frame height in pixels.
            w: Frame width in pixels.

        Returns:
            List of detection dicts with ``'bbox'`` ``[x1, y1, x2, y2]``
            and ``'confidence'`` keys.
        """
        decoded = self._decode_boxes(raw_boxes)
        mask = raw_scores > self.confidence_threshold
        if not np.any(mask):
            return []
        filtered_boxes = decoded[mask]
        filtered_scores = raw_scores[mask]
        detections: list[dict[str, Any]] = []
        for i in range(len(filtered_scores)):
            x1 = filtered_boxes[i][0] * w
            y1 = filtered_boxes[i][1] * h
            x2 = filtered_boxes[i][2] * w
            y2 = filtered_boxes[i][3] * h
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(filtered_scores[i]),
            })
        return detections

    def _postprocess(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply non-maximum suppression to deduplicate detections.

        Args:
            detections: List of detection dicts from ``_decode_and_filter``.

        Returns:
            Filtered list after NMS.
        """
        return self._nms(detections)

    def detect(self, frame: NDArray[np.uint8]) -> list[dict[str, Any]]:
        """Run full face detection on a single RGB frame.

        Args:
            frame: RGB image array ``[H, W, 3]`` uint8.

        Returns:
            List of detection dicts with ``'bbox'`` and ``'confidence'``,
            sorted by descending confidence, after NMS.
        """
        if self.mock:
            return self._mock_detect(frame)
        h, w = frame.shape[:2]
        input_data = self._preprocess(frame)
        result = self._run_inference(input_data)
        if result is None:
            return []
        raw_boxes, raw_scores = result
        detections = self._decode_and_filter(raw_boxes, raw_scores, h, w)
        return self._postprocess(detections)

    def _generate_anchors(self) -> NDArray[np.float32]:
        """Generate SSD anchors for BlazeFace (4 layers, strides [8,16,16,16])."""
        opts = {
            'num_layers': 4,
            'input_height': self.input_size[1],
            'input_width': self.input_size[0],
            'anchor_offset_x': 0.5,
            'anchor_offset_y': 0.5,
            'strides': [8, 16, 16, 16],
            'interpolated_scale_aspect_ratio': 1.0,
        }
        anchors: list[tuple[float, float]] = []
        layer_id = 0
        while layer_id < opts['num_layers']:
            last_same = layer_id
            repeats = 0
            while (last_same < opts['num_layers'] and
                   opts['strides'][last_same] == opts['strides'][layer_id]):
                last_same += 1
                repeats += 2 if opts['interpolated_scale_aspect_ratio'] == 1.0 else 1
            stride = opts['strides'][layer_id]
            fh = opts['input_height'] // stride
            fw = opts['input_width'] // stride
            for y in range(fh):
                yc = (y + opts['anchor_offset_y']) / fh
                for x in range(fw):
                    xc = (x + opts['anchor_offset_x']) / fw
                    for _ in range(repeats):
                        anchors.append((xc, yc))
            layer_id = last_same
        return np.array(anchors, dtype=np.float32)

    def _decode_boxes(self, raw_boxes: NDArray[np.float32]) -> NDArray[np.float64]:
        """Decode raw SSD boxes to [xmin, ymin, xmax, ymax] normalized [0,1]."""
        scale = float(self.input_size[1])
        num_points = raw_boxes.shape[-1] // 2
        boxes = raw_boxes.reshape(-1, num_points, 2).copy() / scale
        boxes[:, 0] += self.anchors  # center offset
        for i in range(2, num_points):
            boxes[:, i] += self.anchors

        center = boxes[:, 0]  # [x_center, y_center]
        half_size = boxes[:, 1] / 2.0  # [w/2, h/2]

        result = np.zeros((raw_boxes.shape[0], 4))
        result[:, 0] = center[:, 0] - half_size[:, 0]
        result[:, 1] = center[:, 1] - half_size[:, 1]
        result[:, 2] = center[:, 0] + half_size[:, 0]
        result[:, 3] = center[:, 1] + half_size[:, 1]
        return result

    def _nms(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Greedy non-maximum suppression on detections.

        Args:
            detections: Candidate detections sorted by confidence descending.

        Returns:
            Detections surviving the IoU threshold.
        """
        if len(detections) <= 1:
            return detections
        detections.sort(key=operator.itemgetter('confidence'), reverse=True)
        keep: list[dict[str, Any]] = []
        for det in detections:
            should_keep = True
            for kept in keep:
                if self._iou(det['bbox'], kept['bbox']) > self.iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(det)
        return keep

    @staticmethod
    def _iou(box1: list[float], box2: list[float]) -> float:
        """Compute intersection-over-union of two axis-aligned boxes.

        Args:
            box1: ``[x1, y1, x2, y2]``.
            box2: ``[x1, y1, x2, y2]``.

        Returns:
            IoU value in ``[0.0, 1.0]``.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _mock_detect(self, frame: NDArray[np.uint8]) -> list[dict[str, Any]]:
        """Mock detection: returns a centered face bbox for testing."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        face_w = int(w * 0.3)
        face_h = int(h * 0.4)
        return [{'bbox': [cx - face_w//2, cy - face_h//2,
                          cx + face_w//2, cy + face_h//2],
                 'confidence': 0.95}]

    def get_best_detection(self, frame: NDArray[np.uint8]) -> dict[str, Any] | None:
        """Detect and return the highest-confidence face, or None."""
        dets = self.detect(frame)
        if not dets:
            return None
        return dets[0]
