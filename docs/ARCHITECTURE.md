# DMS Application Architecture & Specification
## FRDM-IMX93 Driver Monitoring System - Multi-Model Pipeline

**Document Version**: 1.0  
**Date**: 2026-05-06  
**Target**: i.MX93 FRDM (Cortex-A55 + Ethos-U65 NPU)  
**Regulatory Targets**: EU DDAW (2021/1341), EU ADDW (2023/2590), Euro NCAP Driver Engagement v0.9

---

## 1. Executive Summary

**Reference Application**: This architecture is based on the DMS baseline-2 application developed at `C:\Users\sefa.ocakli\Documents\Antigravity\DMS_NXP_App\` which uses MediaPipe FaceLandmarker + YOLOv4-Tiny smoking/calling + blendshape gaze estimation + 17-zone cockpit projection. Key patterns reused: gaze weights (yaw=32, pitch=45), PERCLOS 60s sliding window, adaptive EMA stabilization, 9-point calibration, behavioral thresholds, and warning escalation.

This document specifies a complete Driver Monitoring System (DMS) application for the NXP FRDM-IMX93 development board. The system uses a multi-model inference pipeline running on the Ethos-U65 NPU to perform real-time driver face detection, facial landmark extraction, iris tracking, gaze estimation, and behavioral analysis. It outputs a rendered dashboard to HDMI via the Wayland compositor.

### What We Are Building

A real-time DMS that processes 1280x800 YUYV camera frames at 25fps through six sequential ML inference stages plus behavioral analysis and dashboard rendering. The system detects drowsiness (PERCLOS, blink, yawn), distraction (gaze zone classification, head pose), phone use, microsleep, and sleep events. It produces escalating visual and acoustic warnings on the HDMI display.

### Why This Architecture

The regulatory requirements (EU DDAW, EU ADDW, Euro NCAP v0.9) mandate camera-based direct driver monitoring with specific detection latencies, false-positive/negative rates, and warning escalation patterns. The multi-model approach (YOLO head detection, landmark detection, iris detection, gaze estimation) provides finer control over each pipeline stage compared to monolithic solutions like MediaPipe FaceLandmarker, allowing independent optimization of each model for the Ethos-U65 NPU and per-stage timing budget allocation.

### Key Design Decisions

1. **Multi-model over monolithic**: Separate YOLO, landmark, and iris models allow independent NPU optimization and debugging, at the cost of higher total inference latency that must be managed through strict per-frame timing budgets.
2. **Python initial, C++ migration path**: The initial implementation uses Python with `ethosu.Interpreter` for fastest iteration. The pipeline is structured so each stage maps to a C++ class for later migration.
3. **Geometric gaze over learning-based**: Iris-center ratio between eye corners provides deterministic gaze estimation suitable for regulatory testing zone classification, without requiring a trained gaze model.
4. **GStreamer for capture and display**: Proven pipeline on this board (PXP hardware CSC, G2D blit, waylandsink). Frame extraction via appsink, dashboard push via appsrc.
5. **Closed-loop only**: No biometric storage, no facial recognition, no network transmission -- per DDAW Article 5 privacy requirements.
6. **MediaPipe landmark extraction**: Extract TFLite model from MediaPipe face_landmarker.task bundle, Vela-optimize for NPU. Use only 25 of 478 output landmarks. CPU fallback if Vela rejects unsupported ops.
7. **YOLOv8n-face for head detection**: Ultralytics YOLOv8n-face variant pre-trained on WIDERFACE, 320x320 input, direct TFLite export.
8. **Simulated speed for development**: Hardcoded 70 km/h to validate DDAW/ADDW thresholds. CAN bus integration deferred to production.

---

## 2. Regulatory Compliance Matrix

### 2.1 DDAW (EU 2021/1341) - Drowsiness

| Requirement | Specification | Implementation |
|---|---|---|
| Activation speed | >= 70 km/h | Simulated `vehicle_speed = 70` km/h for development; CAN bus for production |
| Detection threshold | KSS >= 8 (mandatory) | PERCLOS > 20% in 60s window + microsleep events >= 2 in 5 min |
| Early warning | KSS >= 7 (optional) | PERCLOS > 10% in 60s window + cumulative blink duration |
| PERCLOS metric | Eye closure > 80% over time | Eye Aspect Ratio (EAR) < 0.10 threshold, 60s sliding window |
| Warning timing | ASAP, max 5 min from activation | Real-time pipeline; warning within 1 frame of detection |
| Visual warning | Steady or flashing | Dashboard icon: amber advisory, red flashing escalation |
| Acoustic warning | 200-8000 Hz, 50-90 dB | WAVE file playback via ALSA (board MQS jack P15 or HDMI audio) |
| Privacy | No biometric storage | All face/iris data in RAM only, cleared each frame |
| Deactivation | Below 70 km/h | Speed-dependent activation flag |

### 2.2 ADDW (EU 2023/2590) - Distraction

| Requirement | Specification | Implementation |
|---|---|---|
| Activation speed | >= 20 km/h (after 1 min calibration) | Speed threshold + 60s calibration timer at startup |
| Distraction zone | Below 30 degrees downward from ocular reference | Zone 3 (lower dashboard) in XML zone definition; gaze pitch > 30 deg triggers |
| High-speed gaze timer | >= 50 km/h: 3.5s nominal, 5.0s with buffer | `gaze_offroad_timer` accumulates; triggers at 3.5s, buffer expires at 5.0s |
| Low-speed gaze timer | >= 20 km/h: 6.0s nominal, 7.5s with buffer | Same timer with different thresholds based on speed band |
| False-negative deadline | 4.0s (high), 6.5s (low) | Pipeline latency must be < 200ms; detection must fire within deadline |
| Saccade tolerance | >= 50 ms | Minimum gaze-on-road duration to reset timer; 50ms debounce |
| Fixation point zones | 14 zones for testing | XML-defined polygons matching Euro NCAP TB 036 test grid |
| Warning cascade | Visual -> Visual+Acoustic -> +Haptic | Three-tier escalation with configurable durations |
| Camera-only | Direct monitoring required | Camera-based gaze + head pose; no steering-wheel-only mode |

### 2.3 Euro NCAP Driver Engagement v0.9 (30-Point Scoring)

| Category | Points | Timing/Threshold | Implementation |
|---|---|---|---|
| Long distraction (Owl) | Up to 3 pts | Gaze away 3-4s | VATS-style cumulative off-road timer |
| Long distraction (Lizard) | Up to 3 pts | Gaze away 3-4s, different direction | Same timer, direction-agnostic |
| Long distraction (Body Lean) | Up to 3 pts | Body/gaze away 3-4s | Head yaw > 45 deg + gaze off-road |
| Short distraction (VATS) | Up to 3 pts | Cumulative 10s off-road in 30s | Rolling 30s window, sum off-road durations |
| Phone use | Up to 3 pts | Same VATS criteria toward phone zone | Phone zone in XML; gaze toward phone zone triggers |
| Microsleep | Up to 3 pts | Eye closure 1-2s | EAR < 0.10 for 1-2s duration |
| Sleep | Up to 3 pts | Eye closure >= 3s | EAR < 0.10 for >= 3s |
| Unresponsive driver | Up to 3 pts | No gaze return 3s after warning, or eyes closed >= 6s | Post-warning gaze tracking + extended eye closure |
| Drowsiness | Up to 3 pts | KSS >= 7, learning up to 10 min | PERCLOS + blink rate + yawn frequency; 10 min baseline |
| Emergency function | Up to 3 pts | <= 5s after distinct warning | Warning escalation to emergency level within 5s |
| TPR requirements | 40-90% by category | Per-category true positive rate | Tuned via thresholds in config.yaml |
| Noise robustness | Age 16-80, skin types 1-6, eyewear | Varied test subjects | YOLO + landmark models trained on diverse datasets |

---

## 3. System Architecture

### 3.1 Block Diagram

```
+---------------------------------------------------------------------+
|                        DMS Application (Python)                      |
|                                                                      |
|  +----------+    +------------+    +-----------+    +-------------+  |
|  | Camera   |    | Frame      |    | Model     |    | Behavioral  |  |
|  | Capture  |--->| Pipeline   |--->| Pipeline  |--->| Analysis    |  |
|  | (GStream)|    | (Preproc)  |    | (NPU x4)  |    | Engine      |  |
|  +----------+    +------------+    +-----------+    +-------------+  |
|       |                                                      |        |
|       |              +------------+                          |        |
|       +------------> | Dashboard  | <------------------------+        |
|                      | Renderer   |                                   |
|                      +------------+                                   |
|                           |                                           |
|                      +------------+                                   |
|                      | Display    |                                   |
|                      | (GStream)  |                                   |
|                      +------------+                                   |
+---------------------------------------------------------------------+
         |                    |                    |
    /dev/video0          Ethos-U65           waylandsink
    (MIPI CSI)           (NPU)              (HDMI via PXP)
```

### 3.2 Pipeline Data Flow

```
Frame N (1280x800 YUYV from /dev/video0)
  |
  v
[Stage 1] YUYV -> RGB Conversion (CPU or PXP)
  |  1280x800 RGB
  v
[Stage 2] YOLO Head Detection (NPU)
  |  Bounding box: [x1, y1, x2, y2, confidence]
  v
[Stage 3] Head Crop + Resize (CPU)
  |  Cropped RGB patch, resized to landmark input resolution
  v
[Stage 4] Landmark Detection (NPU)
  |  25 landmarks: nose(1), left eye(6), right eye(6), mouth(12)
  v
[Stage 5] Eye Crop + Resize (CPU)
  |  Left eye patch + Right eye patch, each at iris model input resolution
  v
[Stage 6] Iris Detection (NPU)
  |  Left iris center (x, y, r) + Right iris center (x, y, r)
  v
[Stage 7] Gaze Estimation (CPU)
  |  Gaze yaw, pitch -> Dashboard zone index
  v
[Stage 8] Behavioral Analysis (CPU)
  |  PERCLOS, blink rate, yawn, drowsiness score, distraction timer
  v
[Stage 9] Dashboard Rendering (CPU + G2D)
  |  1920x1080 BGRx frame with overlays
  v
[Stage 10] Display Output (PXP + waylandsink)
  -> HDMI 1920x1080
```

### 3.3 Threading Model

```
Main Thread: GStreamer pipeline management, event loop
  |
  +-- Capture Thread: appsink callback -> frame queue
  |
  +-- Inference Thread: dequeue frame -> run pipeline stages 2-7
  |                     -> push results to analysis queue
  |
  +-- Analysis Thread: dequeue results -> behavioral analysis
  |                     -> push dashboard data to display queue
  |
  +-- Display Thread: dequeue dashboard data -> compose frame
                      -> push via appsrc
```

Four threads, three bounded queues (max depth 2 each to limit latency):
- `capture_queue`: Raw RGB frames (1280x800, numpy array)
- `inference_queue`: Structured inference results (landmarks, iris, gaze)
- `display_queue`: Dashboard rendering instructions

---

## 4. Model Zoo

### 4.1 YOLOv8n-face Head Detector

| Property | Value |
|---|---|
| Architecture | YOLOv8n-face (Ultralytics, pre-trained on WIDERFACE) |
| Input resolution | 320x320 RGB |
| Input format | NHWC, uint8, no normalization (INT8 quantized model handles it) |
| Output format | [1, N, 6] where N=number of detections, each: [x1, y1, x2, y2, confidence, class] |
| Class mapping | 0=face |
| Framework source | Ultralytics YOLOv8n-face export to TFLite |
| Quantization | INT8 (full integer, required for NPU) |
| Vela target | `--target-ethos-u65-high-end` |
| Target inference | 8-12 ms on Ethos-U65 |

**Conversion Steps**:
1. Obtain YOLOv8n-face pre-trained weights (WIDERFACE dataset)
2. `yolo export model=yolov8n-face.pt format=tflite imgsz=320 int8=True data=face_dataset.yaml`
3. `vela --model yolov8n-face_integer_quant.tflite --output-dir ./models --target-ethos-u65-high-end`
4. Result: `yolov8n-face_integer_quant_vela.tflite`

**Post-processing**: Non-maximum suppression (NMS) on CPU. Only keep class 0 (head) with confidence > 0.5. Select highest-confidence detection.

### 4.2 Face Landmark Detector

| Property | Value |
|---|---|
| Architecture | MediaPipe Face Landmark (extracted from face_landmarker.task bundle) |
| Input resolution | 192x192 RGB (MediaPipe internal) |
| Input format | NHWC, uint8 (INT8 quantized) |
| Output format | [1, 478, 3] -> 478 landmarks with x, y, z; we use only 25 selected indices |
| Landmark set | 25 of 478 MediaPipe points: nose tip(1), left eye(6), right eye(6), mouth(12) |
| Framework source | Extract TFLite from MediaPipe `face_landmarker.task` bundle |
| Quantization | INT8 (re-quantize extracted model) |
| Vela target | `--target-ethos-u65-high-end` (fallback to CPU if unsupported ops) |
| Target inference | 5-8 ms on Ethos-U65 (or 15-20 ms CPU fallback) |

**MediaPipe-to-DMS landmark mapping (25 of 478)**:
```
DMS Index 0:       MediaPipe 1    (Nose tip)
DMS Index 1-2:     MediaPipe 61,293  (Left eyebrow inner/outer)
DMS Index 3-4:     MediaPipe 292,62  (Right eyebrow inner/outer)
DMS Index 5-10:    MediaPipe 33,160,158,133,153,144  (Left eye: outer, upper2, inner, lower2)
DMS Index 11-16:   MediaPipe 362,387,385,263,373,380  (Right eye: outer, upper2, inner, lower2)
DMS Index 17-24:   MediaPipe 61,39,11,0,267,275,281,362 -> remapped to mouth outline (8 points)
```

**Extraction & Conversion Steps**:
1. Extract TFLite model from `face_landmarker.task` bundle:
   ```python
   # The .task file is a flatbuffer containing the TFLite model
   # Use MediaPipe's model_bundle_schema or direct extraction
   import zipfile
   with zipfile.ZipFile('face_landmarker.task', 'r') as z:
       # Extract the TFLite model from the bundle
       z.extractall('extracted_model/')
   ```
2. Re-quantize to INT8 with representative dataset from AR0144 camera
3. `vela --model face_landmark_extracted_int8.tflite --output-dir ./models --target-ethos-u65-high-end`
4. If Vela fails on unsupported ops: run on CPU via `tflite.Interpreter` (no delegate)
5. Post-processing: extract only the 25 mapped indices from 478 outputs, scale to frame coordinates

### 4.3 Iris Detector

| Property | Value |
|---|---|
| Architecture | Lightweight regression CNN (similar to MediaPipe Iris) |
| Input resolution | 64x64 RGB (per eye) |
| Input format | NHWC, uint8 (INT8 quantized) |
| Output format | [1, 5] -> [iris_center_x, iris_center_y, iris_radius, eye_openness_x, eye_openness_y] normalized [0..1] |
| Framework source | Custom model based on MediaPipe Iris architecture, simplified |
| Quantization | INT8 |
| Vela target | `--target-ethos-u65-high-end` |
| Target inference | 2-3 ms on Ethos-U65 per eye (run twice per frame) |

**Conversion Steps**:
1. Extract iris sub-model from MediaPipe Face Mesh or train custom regression model
2. Input: 64x64 eye region crop from landmark-detected eye bounds
3. Convert to TFLite with INT8 quantization
4. `vela --model iris_int8.tflite --output-dir ./models --target-ethos-u65-high-end`

### 4.4 Optional: Smoking/Calling Detector

| Property | Value |
|---|---|
| Architecture | YOLOv4-tiny (matching GoPoint DMS demo pattern) |
| Input resolution | 320x320 RGB |
| Input format | NHWC, uint8 (INT8 quantized) |
| Output format | [1, N, 6]: [x1, y1, x2, y2, confidence, class] |
| Classes | 0=smoking, 1=calling |
| Framework source | GoPoint DMS demo model (yolov4_tiny_smk_call.tflite) |
| Quantization | INT8 |
| Vela target | `--target-ethos-u65-high-end` |
| Target inference | 8-12 ms on Ethos-U65 |

Note: This model is only needed for Euro NCAP phone-use scoring. It can run at reduced frequency (every 3rd frame) to save NPU time.

### 4.5 Model Summary

| Model | Input | Params (est.) | NPU Time (est.) | Vela File |
|---|---|---|---|---|
| YOLOv8n-face | 320x320x3 | ~3M | 8-12 ms | `yolov8n_face_vela.tflite` |
| MediaPipe Landmark | 192x192x3 | ~2M | 5-8 ms (NPU) / 15-20 ms (CPU) | `face_landmark_vela.tflite` or CPU |
| Iris (per eye) | 64x64x3 | ~0.5M | 2-3 ms | `iris_vela.tflite` |
| Smoking/Calling | 320x320x3 | ~6M | 8-12 ms | `yolov4_smk_call_vela.tflite` |

---

## 5. Pipeline Detail

### 5.1 Stage 1: Camera Capture + Color Conversion

**Input**: /dev/video0, YUYV 1280x800 from GStreamer v4l2src  
**Processing**: GStreamer pipeline `v4l2src ! videoconvert ! appsink` with YUYV to RGB conversion  
**Output**: numpy array [1280, 800, 3] uint8 RGB  
**Timing budget**: 4 ms (hardware-assisted via PXP)

```python
# GStreamer capture pipeline
PIPELINE_CAPTURE = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=1280,height=800,framerate=30/1 ! "
    "imxvideoconvert_pxp ! "
    "video/x-raw,format=RGB ! "
    "appsink max-buffers=2 drop=true emit-signals=true"
)
```

### 5.2 Stage 2: YOLO Head Detection

**Input**: Full frame 1280x800 RGB (or downscaled to 320x320 for model input)  
**Pre-processing**: Resize 1280x800 -> 320x320, keep aspect ratio with letterboxing  
**Model**: YOLO Head Detector (NPU)  
**Post-processing**: NMS, select top-1 head detection, scale bbox to original coordinates  
**Output**: head_bbox = [x1, y1, x2, y2] in original 1280x800 coordinates  
**Timing budget**: 12 ms (resize 2ms + NPU 8ms + NMS 2ms)

### 5.3 Stage 3: Head Crop + Resize

**Input**: frame_rgb [1280, 800, 3] + head_bbox [x1, y1, x2, y2]  
**Processing**: Crop head region with 20% padding, resize to 112x112  
**Output**: head_crop [112, 112, 3] uint8 RGB  
**Timing budget**: 1 ms

### 5.4 Stage 4: Landmark Detection

**Input**: head_resized [112, 112, 3] uint8 RGB  
**Model**: Landmark 25-point detector (NPU)  
**Post-processing**: Reshape [50] -> [25, 2], scale to original frame coordinates  
**Output**: landmarks = [(x, y), ...] 25 points in 1280x800 coordinates  
**Timing budget**: 5 ms (NPU 4ms + scale 1ms)

### 5.5 Stage 5: Eye Crop + Resize

**Input**: landmarks (eye points: indices 5-10 left, 11-16 right)  
**Processing**: Compute bounding box from 6 eye landmarks per eye, crop, resize to 64x64  
**Output**: left_eye_crop [64, 64, 3], right_eye_crop [64, 64, 3]  
**Timing budget**: 1 ms

### 5.6 Stage 6: Iris Detection

**Input**: left_eye_crop [64, 64, 3], right_eye_crop [64, 64, 3]  
**Model**: Iris detector (NPU), run twice per frame  
**Post-processing**: Scale iris coordinates from 64x64 back to eye region, then to frame coordinates  
**Output**: left_iris = (cx, cy, r), right_iris = (cx, cy, r) in frame coordinates  
**Timing budget**: 6 ms (2x 3ms NPU)

### 5.7 Stage 7: Gaze Estimation

**Input**: landmarks + iris positions  
**Processing**: Geometric gaze calculation (CPU only, no NPU)  
**Output**: gaze_yaw, gaze_pitch (degrees), gaze_zone (0-16)  
**Timing budget**: 2 ms

(Detailed in Section 6 below)

### 5.8 Stage 8: Behavioral Analysis

**Input**: landmarks + iris + gaze per frame  
**Processing**: Accumulate metrics over sliding windows  
**Output**: drowsiness_score, distraction_state, blink_count, yawn_state, warning_level  
**Timing budget**: 2 ms

(Detailed in Section 7 below)

### 5.9 Stage 9: Dashboard Rendering

**Input**: frame_rgb + all pipeline results + warning level  
**Processing**: OpenCV overlay drawing on frame, resize to 1920x1080  
**Output**: dashboard_frame [1920, 1080, 4] BGRx  
**Timing budget**: 5 ms

(Detailed in Section 8 below)

### 5.10 Stage 10: Display Output

**Input**: dashboard_frame [1920, 1080, 4] BGRx  
**Processing**: Push to GStreamer appsrc -> PXP -> waylandsink  
**Output**: HDMI display at ~25fps  
**Timing budget**: 2 ms (hardware-assisted)

```python
PIPELINE_DISPLAY = (
    "appsrc format=GST_FORMAT_TIME ! "
    "video/x-raw,format=BGRx,width=1920,height=1080,framerate=25/1 ! "
    "imxvideoconvert_pxp ! "
    "waylandsink fullscreen=true"
)
```

---

## 6. Gaze Estimation Architecture

### 6.1 Mathematical Model

The gaze estimation uses geometric computation from iris positions relative to eye corners, producing yaw and pitch angles projected onto a dashboard reference plane.

#### Iris Center Ratio (per eye)

```
For each eye, given:
  - inner_corner: (x_in, y_in)  -- landmark near nose
  - outer_corner: (x_out, y_out) -- landmark near ear
  - iris_center:  (x_iris, y_iris)

Horizontal ratio (iris_x):
  iris_x = (x_iris - x_out) / (x_in - x_out)
  Maps to [-1, +1] where:
    -1.0 = looking fully toward ear (outward)
     0.0 = centered
    +1.0 = looking fully toward nose (inward)

Vertical ratio (iris_y):
  iris_y = (y_iris - y_top) / (y_bottom - y_top)
  Where y_top and y_bottom are from upper/lower eye landmarks
  Maps to [-1, +1] where:
    -1.0 = looking fully up
     0.0 = centered
    +1.0 = looking fully down
```

#### Gaze Angle Computation

```python
# Average both eyes
avg_iris_x = (left_iris_x + right_iris_x) / 2.0
avg_iris_y = (left_iris_y + right_iris_y) / 2.0

# Convert to angles with calibrated weights
YAW_WEIGHT = 32.0   # degrees per unit iris_x
PITCH_WEIGHT = 45.0  # degrees per unit iris_y

raw_yaw = avg_iris_x * YAW_WEIGHT
raw_pitch = avg_iris_y * PITCH_WEIGHT

# Add head pose contribution (from landmarks)
face_center_x = (landmarks[5][0] + landmarks[11][0]) / 2  # mid-eye
head_yaw = (landmarks[0][0] - face_center_x) / face_width * 60.0
head_pitch = (landmarks[0][1] - face_center_y) / face_height * 45.0

# Combined gaze
gaze_yaw = raw_yaw * 0.6 + head_yaw * 0.4   # weighted blend
gaze_pitch = raw_pitch * 0.6 + head_pitch * 0.4
```

### 6.2 Dashboard Zone Projection

The gaze angles are projected onto a reference plane at dashboard distance to determine which zone the driver is looking at.

```python
# Camera intrinsic assumption (calibrated from camera position)
CAMERA_ORIGIN_X = 960   # center of 1920 display
CAMERA_ORIGIN_Y = 540   # center of 1080 display

# Projection (pixels per degree, calibrated)
PPD_YAW = 5.0    # pixels per degree of yaw
PPD_PITCH = 5.0  # pixels per degree of pitch

# Projected gaze point on dashboard reference plane
gaze_pixel_x = CAMERA_ORIGIN_X + gaze_yaw * PPD_YAW
gaze_pixel_y = CAMERA_ORIGIN_Y + gaze_pitch * PPD_PITCH

# Zone classification using polygon test
gaze_point = (gaze_pixel_x, gaze_pixel_y)
zone_id = classify_zone(gaze_point)  # cv2.pointPolygonTest against XML zones
```

### 6.3 Zone Definitions (17 Zones)

Zones defined as polygons in XML, loaded at startup:

| Zone ID | Name | Description |
|---|---|---|
| 0 | ROAD_AHEAD | Forward road view (on-road) |
| 1 | LEFT_MIRROR | Left side mirror |
| 2 | RIGHT_MIRROR | Right side mirror |
| 3 | REAR_MIRROR | Interior rearview mirror |
| 4 | INSTRUMENTS | Instrument cluster |
| 5 | CENTER_STACK | Center infotainment |
| 6 | LOWER_DASH | Below 30 deg (DDAW Area 3) |
| 7 | PHONE_LEFT | Phone position (left hand) |
| 8 | PHONE_RIGHT | Phone position (right hand) |
| 9 | PASSENGER | Passenger seat area |
| 10 | ROOF | Upward |
| 11 | LAP_LEFT | Left lap |
| 12 | LAP_RIGHT | Right lap |
| 13 | LEFT_WINDOW | Left side window |
| 14 | RIGHT_WINDOW | Right side window |
| 15 | UNKNOWN | Unmapped region |
| 16 | NO_FACE | No face detected |

On-road zones: 0, 1, 2, 3 (mirrors are "briefly on-road" in ADDW).
Off-road zones: 4-14.

### 6.4 Calibration

9-point calibration procedure (from reference app):

```
Calibration points:
  1. Center (straight ahead)
  2. Far left
  3. Far right
  4. Far up
  5. Far down
  6. Upper-left
  7. Upper-right
  8. Lower-left
  9. Lower-right

For each point:
  - Driver gazes at known dashboard location for 2 seconds
  - System records average iris_x, iris_y over that period
  - Computes offset and scale corrections

Calibration output:
  yaw_offset, yaw_scale, pitch_offset, pitch_scale

Applied to raw gaze:
  calibrated_yaw = (raw_yaw - yaw_offset) * yaw_scale
  calibrated_pitch = (raw_pitch - pitch_offset) * pitch_scale
```

### 6.5 Stabilization

Adaptive Exponential Moving Average (EMA) to reduce gaze jitter:

```python
VELOCITY_THRESHOLD = 15.0  # deg/s
FIXATION_ALPHA = 0.3       # smooth during fixation
SACCADE_ALPHA = 0.8        # responsive during saccades

velocity = sqrt((gaze_yaw - prev_yaw)**2 + (gaze_pitch - prev_pitch)**2) * fps
alpha = SACCADE_ALPHA if velocity > VELOCITY_THRESHOLD else FIXATION_ALPHA

smooth_yaw = alpha * gaze_yaw + (1 - alpha) * prev_smooth_yaw
smooth_pitch = alpha * gaze_pitch + (1 - alpha) * prev_smooth_pitch
```

---

## 7. Behavioral Detection Algorithms

### 7.1 Eye Aspect Ratio (EAR)

```python
def compute_ear(eye_landmarks):
    """
    Eye Aspect Ratio from 6 eye landmarks.
    Returns float ~0.3 (open) to ~0.05 (closed).
    Landmarks: [outer, upper1, upper2, inner, lower1, lower2]
    """
    v1 = distance(eye_landmarks[1], eye_landmarks[5])
    v2 = distance(eye_landmarks[2], eye_landmarks[4])
    h = distance(eye_landmarks[0], eye_landmarks[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

EAR_CLOSE_THRESHOLD = 0.10
EAR_BLINK_THRESHOLD = 0.19
```

### 7.2 Blink Detection

```python
class BlinkDetector:
    def __init__(self):
        self.ear_history = deque(maxlen=30)  # ~1 second at 25fps
        self.blink_count = 0
        self.in_blink = False
        self.blink_start = None
        self.blink_durations = deque(maxlen=100)

    def update(self, left_ear, right_ear):
        avg_ear = (left_ear + right_ear) / 2.0
        self.ear_history.append((time.monotonic(), avg_ear))

        if avg_ear < EAR_BLINK_THRESHOLD:
            if not self.in_blink:
                self.in_blink = True
                self.blink_start = time.monotonic()
        else:
            if self.in_blink:
                duration = time.monotonic() - self.blink_start
                self.blink_durations.append(duration)
                self.blink_count += 1
                self.in_blink = False

    def blink_rate(self, window_sec=60):
        cutoff = time.monotonic() - window_sec
        recent = sum(1 for t, _ in self.blink_durations if t > cutoff)
        return recent * (60.0 / window_sec)
```

### 7.3 PERCLOS (Percentage of Eye Closure)

```python
class PERCLOSDetector:
    """
    PERCLOS p80: proportion of time eyes are > 80% closed
    over a sliding 60-second window.
    """
    def __init__(self, window_sec=60):
        self.window_sec = window_sec
        self.closure_events = deque()

    def update(self, left_ear, right_ear):
        now = time.monotonic()
        avg_ear = (left_ear + right_ear) / 2.0
        is_closed = avg_ear < EAR_CLOSE_THRESHOLD
        self.closure_events.append((now, is_closed))
        cutoff = now - self.window_sec
        while self.closure_events and self.closure_events[0][0] < cutoff:
            self.closure_events.popleft()

    def perclos(self):
        if not self.closure_events:
            return 0.0
        closed_count = sum(1 for _, c in self.closure_events if c)
        return closed_count / len(self.closure_events)

    def severity(self):
        p = self.perclos()
        if p > 0.20:
            return 'severe'      # KSS >= 8, mandatory warning
        elif p > 0.10:
            return 'mild'        # KSS >= 7, optional early warning
        else:
            return 'none'
```

### 7.4 Yawn Detection

```python
def compute_mouth_aspect_ratio(mouth_landmarks):
    """
    Mouth Aspect Ratio from 8 mouth landmarks.
    Landmarks: [left, upper1, upper2, upper3, right, lower1, lower2, lower3]
    """
    v1 = distance(mouth_landmarks[1], mouth_landmarks[7])
    v2 = distance(mouth_landmarks[2], mouth_landmarks[6])
    v3 = distance(mouth_landmarks[3], mouth_landmarks[5])
    h = distance(mouth_landmarks[0], mouth_landmarks[4])
    mar = (v1 + v2 + v3) / (3.0 * h)
    return mar

YAWN_THRESHOLD = 0.5
YAWN_MIN_DURATION = 0.5  # seconds
```

### 7.5 Microsleep Detection

```python
class MicrosleepDetector:
    """
    Detects eye closures of 1-2 seconds (microsleep)
    and >= 3 seconds (sleep).
    """
    def __init__(self):
        self.eyes_closed_since = None
        self.state = 'open'

    def update(self, avg_ear):
        is_closed = avg_ear < EAR_CLOSE_THRESHOLD
        if is_closed:
            if self.eyes_closed_since is None:
                self.eyes_closed_since = time.monotonic()
            duration = time.monotonic() - self.eyes_closed_since
            if duration >= 3.0:
                self.state = 'sleep'
            elif duration >= 1.0:
                self.state = 'microsleep'
            else:
                self.state = 'closed'
        else:
            self.eyes_closed_since = None
            self.state = 'open'
        return self.state
```

### 7.6 Distraction Classification (ADDW + Euro NCAP VATS)

```python
class DistractionDetector:
    def __init__(self):
        self.offroad_timer = 0.0
        self.last_on_road_time = time.monotonic()
        self.vats_window = deque(maxlen=750)  # 30s * 25fps
        self.last_saccade_end = None
        self.SACCADE_TOLERANCE = 0.050  # 50 ms

    def update(self, gaze_zone, vehicle_speed, dt):
        is_on_road = gaze_zone in (0, 1, 2, 3)
        self.vats_window.append((time.monotonic(), not is_on_road))

        if is_on_road:
            now = time.monotonic()
            if (self.last_saccade_end and
                (now - self.last_saccade_end) < self.SACCADE_TOLERANCE):
                pass  # Ignore brief on-road glances within tolerance
            else:
                self.offroad_timer = 0.0
                self.last_saccade_end = None
        else:
            self.offroad_timer += dt
            self.last_saccade_end = time.monotonic()

        # ADDW thresholds based on speed
        if vehicle_speed >= 50:
            nominal, buffer, fn_deadline = 3.5, 5.0, 4.0
        else:
            nominal, buffer, fn_deadline = 6.0, 7.5, 6.5

        # VATS: cumulative 10s off-road in 30s
        now = time.monotonic()
        cutoff = now - 30.0
        recent_offroad = sum(1 for t, is_off in self.vats_window
                            if t > cutoff and is_off) * (1.0/25.0)
        vats_triggered = recent_offroad >= 10.0

        return {
            'addw_nominal': self.offroad_timer >= nominal,
            'addw_buffer': self.offroad_timer >= buffer,
            'addw_fn_deadline': self.offroad_timer >= fn_deadline,
            'vats_triggered': vats_triggered,
            'vats_cumulative': recent_offroad,
            'continuous_offroad': self.offroad_timer,
        }
```

### 7.7 Drowsiness Scoring (KSS Estimation)

```python
class DrowsinessScorer:
    def __init__(self):
        self.learning_start = time.monotonic()
        self.LEARNING_PERIOD = 600.0  # 10 minutes
        self.health_score = 100.0
        self.RECOVERY_RATE = 0.5
        self.PENALTIES = {
            'blink_slow': 3.0, 'yawn': 5.0, 'microsleep': 15.0,
            'perclos_mild': 8.0, 'perclos_severe': 20.0, 'head_drop': 10.0,
        }

    def update(self, perclos_severity, blink_rate, yawn_active,
               microsleep_state, head_pose):
        if perclos_severity == 'severe':
            self.health_score -= self.PENALTIES['perclos_severe']
        elif perclos_severity == 'mild':
            self.health_score -= self.PENALTIES['perclos_mild']

        if microsleep_state == 'microsleep':
            self.health_score -= self.PENALTIES['microsleep']
        elif microsleep_state == 'sleep':
            self.health_score -= self.PENALTIES['microsleep'] * 2

        if yawn_active:
            self.health_score -= self.PENALTIES['yawn']

        if blink_rate > 25:
            self.health_score -= self.PENALTIES['blink_slow']

        if abs(head_pose.get('pitch', 0)) > 25:
            self.health_score -= self.PENALTIES['head_drop']

        self.health_score += self.RECOVERY_RATE
        self.health_score = max(0, min(100, self.health_score))

        # Map to KSS
        if self.health_score >= 80: return 1
        elif self.health_score >= 60: return 3
        elif self.health_score >= 40: return 5
        elif self.health_score >= 25: return 7
        else: return 8
```

### 7.8 Warning Escalation

```python
class WarningManager:
    LEVELS = ['none', 'advisory', 'escalating', 'intervention', 'emergency'

]

    def __init__(self):
        self.level = 'none'
        self.level_start = None
        self.escalation_durations = {
            'advisory': 3.0, 'escalating': 5.0, 'intervention': 5.0,
        }

    def update(self, kss, distraction, microsleep, drowsiness_severity):
        now = time.monotonic()
        target_level = 'none'

        if kss >= 8 or microsleep == 'sleep':
            target_level = 'emergency'
        elif kss >= 7 or microsleep == 'microsleep':
            target_level = 'escalating'
        elif distraction['addw_nominal'] or distraction['vats_triggered']:
            target_level = 'escalating'
        elif distraction['addw_buffer']:
            target_level = 'intervention'
        elif drowsiness_severity == 'mild':
            target_level = 'advisory'

        level_idx = self.LEVELS.index(self.level)
        target_idx = self.LEVELS.index(target_level)
        if target_idx > level_idx:
            if self.level_start is None:
                self.level_start = now
            duration = now - self.level_start
            max_dur = self.escalation_durations.get(self.level, 0)
            if duration >= max_dur:
                self.level = target_level
                self.level_start = now
        elif target_idx < level_idx:
            self.level = target_level
            self.level_start = now

        return self.level
```

---

## 8. Display Dashboard Design

### 8.1 Layout (1920x1080)

```
+-------------------------------------------+-------------------+
|                                            |   DMS Status      |
|                                            |   Panel           |
|                                            |                   |
|                                            |  [KSS: 3 ALERT]  |
|        Camera Feed                         |  [PERCLOS: 5%]   |
|        (scaled to fit)                     |  [Blink: 12/m]   |
|                                            |  [Gaze: ROAD]    |
|                                            |  [Zone: 0]       |
|                                            |  [FPS: 25]       |
|                                            |  [Head: detected]|
|                                            |                   |
|                                            |  Warning Area     |
|                                            |  [=== GREEN ===]  |
|                                            |                   |
|                                            |  Landmarks View   |
|                                            |  [mini face viz]  |
+-------------------------------------------+-------------------+
|  Timeline Bar: gaze history, blink events, PERCLOS trend       |
+----------------------------------------------------------------+
```

### 8.2 Rendering Pipeline

```python
class DashboardRenderer:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.frame = np.zeros((height, width, 4), dtype=np.uint8)

    def render(self, camera_frame, results):
        self.frame[:] = 0
        # Camera feed (left 1440x1080)
        cam_resized = cv2.resize(camera_frame, (1440, 1080))
        self.frame[:1080, :1440, :3] = cam_resized
        # Status panel (right 480x1080)
        self._draw_status_panel(results)
        # Landmarks overlay
        self._draw_landmarks(results['landmarks'])
        # Gaze ray
        self._draw_gaze_ray(results['gaze_yaw'], results['gaze_pitch'])
        # Warning indicator
        self._draw_warning(results['warning_level'])
        # Timeline bar (bottom 80px)
        self._draw_timeline(results)
        return self.frame
```

### 8.3 Warning Visuals

| Warning Level | Color | Behavior |
|---|---|---|
| none | Green | Solid green bar in status panel |
| advisory | Amber/Yellow | Pulsing amber bar, text message |
| escalating | Orange | Flashing orange bar + icon, text + acoustic |
| intervention | Red | Fast flashing red bar, acoustic + haptic indicator |
| emergency | Red + White | Alternating red/white flash, loud acoustic |

### 8.4 Display GStreamer Pipeline

```python
PIPELINE_DISPLAY = (
    "appsrc format=GST_FORMAT_TIME block=true max-buffers=2 ! "
    "video/x-raw,format=BGRx,width=1920,height=1080,framerate=25/1 ! "
    "imxvideoconvert_pxp ! "
    "waylandsink fullscreen=true sync=false"
)
```

---

## 9. Performance Budget

Target: **25 fps = 40 ms per frame**

| Stage | Processing | Time (ms) | Notes |
|---|---|---|---|
| 1. Capture + YUYV->RGB | GStreamer + PXP HW | 4 | Hardware color conversion |
| 2. YOLO Head Detection | Resize + NPU + NMS | 12 | Largest model |
| 3. Head Crop + Resize | CPU cv2.resize | 1 | Small crop region |
| 4. Landmark Detection | NPU (or CPU) | 5-8 (NPU) / 15-20 (CPU) | MediaPipe extracted model, 192x192 input |
| 5. Eye Crop + Resize | CPU cv2.resize | 1 | Two small crops |
| 6. Iris Detection (x2) | NPU | 6 | Two passes, 3ms each |
| 7. Gaze Estimation | CPU geometric | 2 | Math only, no ML |
| 8. Behavioral Analysis | CPU accumulation | 2 | Sliding window updates |
| 9. Dashboard Rendering | CPU OpenCV drawing | 5 | 1920x1080 compositing |
| 10. Display Push | PXP + waylandsink | 2 | Hardware path |
| **Total** | | **40** (NPU all) / **42** (CPU landmark) | **Target 25fps** |

### NPU Utilization

| Model | NPU Time | Runs/Frame | Total NPU |
|---|---|---|---|
| YOLO Head | 8 ms | 1 | 8 ms |
| Landmark | 8 ms | 1 | 8 ms (NPU) or 0 ms (CPU path) |
| Iris | 3 ms | 2 | 6 ms |
| **Total NPU** | | | **22 ms / 40 ms = 55%** |

If landmark runs on CPU (Vela incompatibility): NPU = 14 ms (35%), CPU adds 15-20 ms for landmark.
In that case, total frame = 14ms NPU + 20ms CPU landmark + 8ms other = ~42ms → ~24fps (close to target).
Mitigation: Run landmark inference on CPU in parallel with iris on NPU (different threads).

This leaves room for the optional Smoking/Calling detector (8ms every 3rd frame = ~2.7ms amortized).

### Mitigation if Budget Exceeds

1. **Frame skip for display**: Run inference at 25fps but render at 15fps
2. **Smoking/Calling at reduced rate**: Run every 3rd-5th frame
3. **Reduce YOLO input**: Use 256x256 instead of 320x320
4. **Single eye iris**: Run iris on one eye only, infer other (saves 3ms)
5. **Landmark reuse**: If head position changes <5px, skip Stage 4

---

## 10. Model Conversion & Deployment

### 10.1 Universal Workflow

For every model:
1. **Source**: PyTorch / Keras / ONNX model weights
2. **Export to TFLite**: `tf.lite.TFLiteConverter` from SavedModel or ONNX
3. **INT8 Quantization**: Representative dataset of 100-500 images, full integer quantization
4. **Vela Optimization**: `vela --target-ethos-u65-high-end` for Ethos-U65 NPU
5. **Deploy**: Copy `_vela.tflite` to board `/opt/dms/models/`

### 10.2 Per-Model Conversion

#### YOLOv8n-face Head Detector
```bash
# From Ultralytics YOLOv8-face pre-trained weights
yolo export model=yolov8n-face.pt format=tflite imgsz=320 int8=True data=face_dataset.yaml
vela --model yolov8n-face_integer_quant.tflite --output-dir ./models --target-ethos-u65-high-end
```

#### MediaPipe Face Landmark (extracted from .task bundle)
```bash
# Step 1: Extract TFLite from face_landmarker.task
python3 extract_mediapipe_model.py --task face_landmarker.task --output face_landmark.tflite
# Step 2: Re-quantize to INT8 with AR0144 camera representative dataset
python3 quantize_model.py --model face_landmark.tflite --calib_images ./calibration_images/ --output face_landmark_int8.tflite
# Step 3: Vela optimize (may fail on unsupported ops - that's OK, use CPU fallback)
vela --model face_landmark_int8.tflite --output-dir ./models --target-ethos-u65-high-end
# If Vela succeeds: deploy face_landmark_int8_vela.tflite to NPU
# If Vela fails: deploy face_landmark_int8.tflite to CPU via tflite.Interpreter
```

#### Iris Detector
```bash
python3 convert_iris.py --model iris_detector.h5 --output iris_detector.tflite --quantize int8
vela --model iris_detector.tflite --output-dir ./models --target-ethos-u65-high-end
```

#### Smoking/Calling (from GoPoint demo)
```bash
# Already available: /opt/gopoint-apps/downloads/yolov4_tiny_smk_call.tflite
vela --model yolov4_tiny_smk_call.tflite --output-dir ./models --target-ethos-u65-high-end
```

### 10.3 Quantization Representative Dataset

Each model needs a representative dataset of real camera frames for INT8 calibration:
- Capture 200-500 frames from the actual AR0144 camera on the FRDM board
- Cover: different lighting (1-100000 lux), diverse faces, eyewear, head angles
- Save as individual crops at each model's input resolution
- Use `tf.data.Dataset` generator for the converter

### 10.4 Deployment to Board

```bash
scp models/*_vela.tflite root@192.168.1.100:/opt/dms/models/
python3 -c "import ethosu; m = ethosu.Interpreter('/opt/dms/models/yolov8n_head_vela.tflite'); print('OK')"
```

---

## 11. Configuration & Calibration

### 11.1 Runtime Configuration (config.yaml)

```yaml
dms:
  camera:
    device: "/dev/video0"
    width: 1280
    height: 800
    format: "YUYV"
    target_fps: 25

  vehicle:
    speed_source: "simulate"  # "can", "simulate", "constant"
    simulated_speed: 70  # km/h (for development)
    ddaw_activation_speed: 70
    addw_activation_speed: 20

  models:
    head_detector: "/opt/dms/models/yolov8n_face_vela.tflite"
    landmark: "/opt/dms/models/face_landmark_vela.tflite"
    landmark_fallback: "/opt/dms/models/face_landmark.tflite"  # CPU if Vela fails
    iris: "/opt/dms/models/iris_detector_vela.tflite"
    smoking_calling: "/opt/dms/models/yolov4_smk_call_vela.tflite"
    run_smoking_calling: true
    smoking_calling_every_n: 3

  gaze:
    yaw_weight: 32.0
    pitch_weight: 45.0
    ppd_yaw: 5.0
    ppd_pitch: 5.0
    stabilization_alpha_fixation: 0.3
    stabilization_alpha_saccade: 0.8
    saccade_velocity_threshold: 15.0
    zones_file: "/opt/dms/config/zones.xml"

  thresholds:
    ear_close: 0.10
    ear_blink: 0.19
    yawn_mar: 0.5
    yawn_min_duration: 0.5
    head_yaw_max: 25.0
    head_pitch_max: 20.0
    perclos_window_sec: 60
    perclos_mild: 0.10
    perclos_severe: 0.20

  timing:
    addw_high_speed_nominal: 3.5
    addw_high_speed_buffer: 5.0
    addw_high_speed_fn_deadline: 4.0
    addw_low_speed_nominal: 6.0
    addw_low_speed_buffer: 7.5
    addw_low_speed_fn_deadline: 6.5
    saccade_tolerance_ms: 50
    vats_window_sec: 30
    vats_threshold_sec: 10.0

  warnings:
    escalation_advisory_sec: 3.0
    escalation_escalating_sec: 5.0
    escalation_intervention_sec: 5.0
    acoustic_file_advisory: "/opt/dms/sounds/advisory.wav"
    acoustic_file_escalating: "/opt/dms/sounds/escalating.wav"
    acoustic_file_emergency: "/opt/dms/sounds/emergency.wav"

  display:
    width: 1920
    height: 1080
    fullscreen: true
    show_landmarks: true
    show_gaze_ray: true
    show_zones: false

  calibration:
    enabled: true
    auto_start: true
    points: 9
    duration_per_point: 2.0
    learning_period_sec: 600

  logging:
    level: "INFO"
    file: "/tmp/dms.log"
    metrics_csv: "/tmp/dms_metrics.csv"
```

### 11.2 Zone Definition File (zones.xml)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<zones display_width="1920" display_height="1080">
  <zone id="0" name="ROAD_AHEAD" type="on_road">
    <polygon points="660,100 1260,100 1260,900 660,900"/>
  </zone>
  <zone id="1" name="LEFT_MIRROR" type="on_road">
    <polygon points="0,200 300,200 300,600 0,600"/>
  </zone>
  <zone id="2" name="RIGHT_MIRROR" type="on_road">
    <polygon points="1620,200 1920,200 1920,600 1620,600"/>
  </zone>
  <zone id="3" name="REAR_MIRROR" type="on_road">
    <polygon points="760,0 1160,0 1160,150 760,150"/>
  </zone>
  <zone id="6" name="LOWER_DASH" type="off_road">
    <polygon points="660,900 1260,900 1260,1080 660,1080"/>
  </zone>
  <zone id="7" name="PHONE_LEFT" type="off_road">
    <polygon points="300,600 660,600 660,900 300,900"/>
  </zone>
  <zone id="8" name="PHONE_RIGHT" type="off_road">
    <polygon points="1260,600 1620,600 1620,900 1260,900"/>
  </zone>
</zones>
```

---

## 12. File Structure

### 12.1 Project Directory Layout (on board at /opt/dms/)

```
/opt/dms/
|-- dms_app.py                         # Main entry point
|-- config/
|   |-- config.yaml                    # Runtime configuration
|   |-- zones.xml                      # Gaze zone polygon definitions
|   |-- sounds/
|       |-- advisory.wav               # Advisory acoustic warning
|       |-- escalating.wav             # Escalating acoustic warning
|       |-- emergency.wav              # Emergency acoustic warning
|
|-- models/
|   |-- yolov8n_face_vela.tflite       # YOLOv8n-face head detector (NPU)
|   |-- yolov8n_face.tflite            # YOLOv8n-face head detector (CPU fallback)
|   |-- face_landmark_vela.tflite      # MediaPipe landmark extracted (NPU)
|   |-- face_landmark.tflite           # MediaPipe landmark extracted (CPU fallback)
|   |-- iris_detector_vela.tflite      # Iris detector (NPU)
|   |-- iris_detector.tflite           # Iris detector (CPU fallback)
|   |-- yolov4_smk_call_vela.tflite    # Smoking/calling (NPU, optional)
|   |-- yolov4_smk_call.tflite         # Smoking/calling (CPU fallback)
|
|-- src/
|   |-- __init__.py
|   |-- pipeline.py                    # Main pipeline orchestrator
|   |-- capture.py                     # GStreamer camera capture
|   |-- display.py                     # GStreamer display output
|   |-- head_detector.py               # YOLO head detection stage
|   |-- landmark_detector.py           # Landmark detection stage
|   |-- iris_detector.py               # Iris detection stage
|   |-- gaze_estimator.py              # Gaze estimation (geometric)
|   |-- behavioral/
|   |   |-- __init__.py
|   |   |-- perclos.py                 # PERCLOS drowsiness
|   |   |-- blink.py                   # Blink detection
|   |   |-- yawn.py                    # Yawn detection
|   |   |-- microsleep.py              # Microsleep/sleep detection
|   |   |-- distraction.py             # ADDW + VATS distraction
|   |   |-- drowsiness.py              # KSS scoring
|   |   |-- warning.py                 # Warning escalation manager
|   |-- dashboard/
|   |   |-- __init__.py
|   |   |-- renderer.py               # Dashboard frame composition
|   |   |-- overlays.py               # Landmark, gaze, zone overlays
|   |   |-- warnings.py               # Warning visual rendering
|   |-- calibration/
|   |   |-- __init__.py
|   |   |-- gaze_calibration.py        # 9-point gaze calibration
|   |   |-- baseline.py               # Driver baseline learning
|   |-- utils/
|       |-- __init__.py
|       |-- npu.py                     # NPU interpreter wrapper
|       |-- image.py                   # Crop, resize, color conversion
|       |-- geometry.py                # Distance, angle, polygon test
|       |-- config.py                  # YAML config loader
|       |-- zone_loader.py             # XML zone polygon loader
|
|-- scripts/
|   |-- capture_calibration.py         # Capture calibration images
|   |-- benchmark_models.py           # Per-model NPU timing benchmark
|   |-- test_pipeline.py              # End-to-end pipeline test
|   |-- generate_zones.py             # Zone XML generator tool
|
|-- systemd/
    |-- dms.service                    # systemd unit file for auto-start
```

### 12.2 Development Host Layout

```
d:/imx93_projects/dms/
|-- README.md
|-- docs/
|   |-- ARCHITECTURE.md               # This document
|   |-- MODEL_CONVERSION.md           # Detailed model conversion guide
|   |-- REGULATORY_MAPPING.md         # Compliance traceability
|   |-- CALIBRATION_GUIDE.md          # Calibration procedure
|
|-- model_training/
|   |-- head_detector/
|   |   |-- train.py                  # YOLOv8 head detector training
|   |   |-- export_tflite.py          # Export to TFLite + quantize
|   |   |-- dataset/                  # Head detection dataset
|   |-- landmark_25pt/
|   |   |-- train.py                  # PFLD-25 training
|   |   |-- export_tflite.py          # Export to TFLite + quantize
|   |   |-- dataset/                  # 300W-LP / WFLW dataset
|   |-- iris/
|       |-- train.py                  # Iris regression training
|       |-- export_tflite.py          # Export to TFLite + quantize
|       |-- dataset/                  # Iris annotation dataset
|
|-- calibration_images/               # For INT8 representative dataset
|-- deploy/
    |-- deploy_to_board.sh            # SCP models + code to board
    |-- install_service.sh            # Install systemd service
```

---

## 13. Verification Plan

### 13.1 Model-Level Tests

| Test | Method | Pass Criteria |
|---|---|---|
| YOLO head detection accuracy | Test on 100 labeled frames with head present | mAP > 0.85, recall > 0.95 |
| YOLO no-face rejection | Test on 50 frames without face | FP rate < 5% |
| Landmark accuracy | Test on 300W test set, normalize error | NME < 8% of inter-ocular distance |
| Iris detection accuracy | Test on labeled iris dataset | Center error < 5% of eye width |
| NPU inference timing | Benchmark 1000 inferences per model | Within timing budget (Section 9) |
| Vela compatibility | Run each _vela.tflite through ethosu.Interpreter | No errors, outputs match CPU model |

### 13.2 Pipeline-Level Tests

| Test | Method | Pass Criteria |
|---|---|---|
| End-to-end latency | Measure frame capture to display output | < 40ms (25fps) |
| Detection consistency | Process 1000 frames, count face detection drops | Drop rate < 2% |
| Gaze zone accuracy | Display calibration points, verify zone classification | > 90% correct zone |
| Iris tracking stability | Measure gaze jitter over static fixation | Std dev < 2 degrees |

### 13.3 Regulatory Scenario Tests

| Test | Method | Pass Criteria |
|---|---|---|
| DDAW PERCLOS detection | Simulate gradual eye closure over 60s | KSS >= 8 warning within 5 min |
| ADDW gaze-off-road detection | Look away for 3.5s at 70 km/h | Warning triggers within 4.0s |
| ADDW low-speed test | Look away for 6.0s at 30 km/h | Warning triggers within 6.5s |
| NCAP microsleep | Close eyes for 1.5s | Detected as microsleep event |
| NCAP sleep | Close eyes for 4.0s | Detected as sleep event |
| NCAP VATS | Cumulative 10s off-road in 30s | VATS triggered, phone-use category |
| NCAP unresponsive | Ignore warning for 3s | Unresponsive detected |
| Warning escalation | Trigger each level | Visual + acoustic cascade follows spec |
| Saccade tolerance | Brief 50ms glance back | Does NOT reset off-road timer |

### 13.4 Robustness Tests

| Test | Method | Pass Criteria |
|---|---|---|
| Low light (1 lux) | Test in dark room | Face detected > 80% of frames |
| Bright light (100k lux) | Test with direct sun | Face detected > 80% of frames |
| Glasses | Test with 5 subjects wearing glasses | Landmark accuracy within 10% of bare-face |
| Sunglasses | Test with dark sunglasses | Iris may fail; head pose fallback activates |
| Face mask | Test with surgical mask | Lower-face landmarks degraded; upper-face only mode |
| Facial hair | Test with beard/mustache | No significant accuracy drop |
| Age range | Test with subjects 16-80 | Consistent detection across ages |
| Skin types 1-6 | Fitzpatrick scale | No accuracy variation by skin tone |

### 13.5 Performance Regression Tests

| Test | Method | Pass Criteria |
|---|---|---|
| Sustained FPS | Run for 30 minutes, log FPS | Average >= 23 fps, min >= 20 fps |
| Memory stability | Monitor RSS over 30 minutes | No growth > 10 MB |
| NPU stability | Monitor ethosu device over 30 minutes | No NPU errors in dmesg |
| Thermal | Monitor SoC temperature | < 85C sustained |

---

## Appendix A: NPU Interpreter Wrapper

```python
# src/utils/npu.py
import ethosu.interpreter as ethosu
import tflite_runtime.interpreter as tflite
import numpy as np

class NPUModel:
    """Unified wrapper for NPU inference with CPU fallback."""

    def __init__(self, model_path, use_npu=True):
        self.use_npu = use_npu
        if use_npu:
            try:
                self.interpreter = ethosu.Interpreter(model_path)
                self.api = 'ethosu'
            except Exception:
                delegate = tflite.load_delegate("/usr/lib/libethosu_delegate.so")
                self.interpreter = tflite.Interpreter(
                    model_path=model_path, experimental_delegates=[delegate])
                self.interpreter.allocate_tensors()
                self.api = 'tflite_delegate'
        else:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.api = 'tflite_cpu'

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def predict(self, input_data):
        if input_data.shape != tuple(self.input_shape):
            raise ValueError(f"Expected {self.input_shape}, got {input_data.shape}")

        if self.api == 'ethosu':
            self.interpreter.set_input(0, input_data)
            self.interpreter.invoke()
            outputs = [self.interpreter.get_output(d['index'])[0]
                      for d in self.output_details]
        else:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            outputs = [self.interpreter.get_tensor(d['index'])[0]
                      for d in self.output_details]
        return outputs
```

## Appendix B: Systemd Service

```ini
# /etc/systemd/system/dms.service
[Unit]
Description=DMS Driver Monitoring System
After=weston.service camera-setup.service
Requires=weston.service camera-setup.service

[Service]
Type=simple
User=root
Environment=XDG_RUNTIME_DIR=/run/user/0
Environment=WAYLAND_DISPLAY=wayland-0
ExecStart=/usr/bin/python3 /opt/dms/dms_app.py --config /opt/dms/config/config.yaml
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## Appendix C: Key Thresholds Quick Reference

| Parameter | Value | Source |
|---|---|---|
| PERCLOS severe | > 20% in 60s | DDAW KSS >= 8 |
| PERCLOS mild | > 10% in 60s | NCAP KSS >= 7 |
| EAR close threshold | 0.10 | Reference app |
| EAR blink threshold | 0.19 | Reference app |
| MAR yawn threshold | 0.50 | Reference app |
| Head yaw limit | 25 deg | Reference app |
| Head pitch limit | 20 deg | Reference app |
| ADDW high-speed nominal | 3.5s | ADDW >= 50 km/h |
| ADDW high-speed buffer | 5.0s | ADDW >= 50 km/h |
| ADDW low-speed nominal | 6.0s | ADDW >= 20 km/h |
| ADDW low-speed buffer | 7.5s | ADDW >= 20 km/h |
| Saccade tolerance | 50 ms | ADDW |
| VATS window | 30s | Euro NCAP |
| VATS threshold | 10s cumulative | Euro NCAP |
| Microsleep duration | 1-2s | Euro NCAP |
| Sleep duration | >= 3s | Euro NCAP |
| Unresponsive timeout | 3s after warning | Euro NCAP |
| Emergency function | <= 5s after warning | Euro NCAP |
| DDAW activation | >= 70 km/h | DDAW |
| ADDW activation | >= 20 km/h | ADDW |
| Calibration period | 60s | ADDW |
| Learning period | 10 min | Euro NCAP |

---

*End of DMS Architecture & Specification Document*
