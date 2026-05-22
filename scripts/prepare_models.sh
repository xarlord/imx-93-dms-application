#!/bin/bash
# prepare_models.sh - Download and convert ML models for DMS NPU inference
#
# This script:
#   1. Downloads pre-trained models (YOLOv8n-face, MediaPipe face_landmarker)
#   2. Exports to TFLite format
#   3. Quantizes to INT8
#   4. Optimizes with Vela for Ethos-U65
#
# Prerequisites:
#   - Python 3.10+ with: ultralytics, tensorflow, onnx, numpy
#   - Vela compiler: pip install ethos-u-vela
#   - Calibration images from AR0144 camera (200+ frames)
#
# Usage:
#   ./prepare_models.sh [--download-only] [--vela-only] [--model MODEL]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models"
CALIB_DIR="${SCRIPT_DIR}/../calibration_images"
mkdir -p "${MODEL_DIR}"

DOWNLOAD_ONLY=0
VELA_ONLY=0
TARGET_MODEL="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --download-only) DOWNLOAD_ONLY=1; shift ;;
        --vela-only) VELA_ONLY=1; shift ;;
        --model) TARGET_MODEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== DMS Model Preparation ==="
echo "Model dir: ${MODEL_DIR}"
echo "Target: ${TARGET_MODEL}"
echo ""

# ============================================================
# Model 1: YOLOv8n-face Head Detector
# ============================================================
if [[ "${TARGET_MODEL}" == "all" || "${TARGET_MODEL}" == "yolo" ]]; then
    echo "--- YOLOv8n-face Head Detector ---"

    if [[ ${VELA_ONLY} -eq 0 ]]; then
        echo "[1] Exporting YOLOv8n-face to TFLite..."
        python3 -c "
from ultralytics import YOLO
# Download pre-trained YOLOv8n-face
model = YOLO('yolov8n-face.pt')
# Export to TFLite with INT8 quantization
model.export(
    format='tflite',
    imgsz=320,
    int8=True,
    data='face_dataset.yaml' if __import__('os').path.exists('face_dataset.yaml') else None,
)
print('Export complete: yolov8n-face_integer_quant.tflite')
" || echo "WARNING: YOLO export failed. Pre-trained weights may not be available."
        echo ""
    fi

    echo "[2] Vela optimization for Ethos-U65..."
    if [[ -f "yolov8n-face_integer_quant.tflite" ]]; then
        vela --model yolov8n-face_integer_quant.tflite \
              --output-dir "${MODEL_DIR}" \
              --target-ethos-u65-high-end \
              --output-name yolov8n_face_vela
        cp yolov8n-face_integer_quant.tflite "${MODEL_DIR}/yolov8n_face.tflite"
        echo "  -> ${MODEL_DIR}/yolov8n_face_vela.tflite"
        echo "  -> ${MODEL_DIR}/yolov8n_face.tflite (CPU fallback)"
    else
        echo "  SKIPPED: TFLite model not found"
    fi
    echo ""
fi

# ============================================================
# Model 2: MediaPipe Face Landmark (extracted from .task bundle)
# ============================================================
if [[ "${TARGET_MODEL}" == "all" || "${TARGET_MODEL}" == "landmark" ]]; then
    echo "--- MediaPipe Face Landmark ---"

    if [[ ${VELA_ONLY} -eq 0 ]]; then
        echo "[1] Extracting TFLite from MediaPipe face_landmarker.task..."
        python3 -c "
import mediapipe as mp
import os

# Download face_landmarker.task if not present
task_file = 'face_landmarker.task'
if not os.path.exists(task_file):
    import urllib.request
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    print(f'Downloading {url}...')
    urllib.request.urlretrieve(url, task_file)
    print('Download complete')

# Extract TFLite model from the task bundle
# The .task file is a FlatBuffer containing the TFLite model
import zipfile
if zipfile.is_zipfile(task_file):
    with zipfile.ZipFile(task_file) as z:
        for name in z.namelist():
            print(f'  Found: {name}')
            if name.endswith('.tflite') or name.endswith('.bin'):
                z.extract(name, 'extracted/')
else:
    # It may be a raw FlatBuffer - use MediaPipe's model bundle tools
    print('  Task file is not a ZIP, attempting direct extraction...')
    import struct
    with open(task_file, 'rb') as f:
        data = f.read()
    # Search for TFLite magic bytes
    tfl_magic = b'\\x54\\x46\\x4c\\x33'  # TFL3
    idx = data.find(tfl_magic)
    if idx >= 0:
        print(f'  Found TFLite header at offset {idx}')
        # Write the TFLite portion
        with open('face_landmark_extracted.tflite', 'wb') as out:
            out.write(data[idx:])
        print('  Extracted: face_landmark_extracted.tflite')
    else:
        print('  Could not find TFLite header in task file')
print('Extraction complete')
" || echo "WARNING: MediaPipe extraction failed."
        echo ""
    fi

    echo "[2] Vela optimization (may fail - CPU fallback supported)..."
    for model_file in face_landmark_extracted.tflite face_landmark.tflite; do
        if [[ -f "${model_file}" ]]; then
            vela --model "${model_file}" \
                  --output-dir "${MODEL_DIR}" \
                  --target-ethos-u65-high-end \
                  --output-name face_landmark_vela && \
            echo "  -> ${MODEL_DIR}/face_landmark_vela.tflite (NPU)" || \
            echo "  Vela failed (expected for some MediaPipe ops) - using CPU fallback"
            cp "${model_file}" "${MODEL_DIR}/face_landmark.tflite"
            echo "  -> ${MODEL_DIR}/face_landmark.tflite (CPU fallback)"
            break
        fi
    done
    echo ""
fi

# ============================================================
# Model 3: Iris Detector
# ============================================================
if [[ "${TARGET_MODEL}" == "all" || "${TARGET_MODEL}" == "iris" ]]; then
    echo "--- Iris Detector ---"
    echo "[1] Iris model preparation..."
    echo "  The iris detector model should be trained separately."
    echo "  See: dms/model_training/iris/train.py"
    echo "  For initial testing, mock mode will be used."
    echo ""
fi

# ============================================================
# Summary
# ============================================================
echo "=== Model Preparation Summary ==="
echo ""
echo "Model files in ${MODEL_DIR}/:"
ls -la "${MODEL_DIR}/"*.tflite 2>/dev/null || echo "  (no models yet)"
echo ""
echo "Deploy to board:"
echo "  scp ${MODEL_DIR}/*.tflite root@192.168.1.100:/opt/dms/models/"
echo ""
echo "Test on board:"
echo "  python3 /opt/dms/scripts/test_board_integration.py --check-models"
