#!/bin/bash
# YOLO11l-obb DeepX (.dxnn) Predict - CLI
#
# Runs inference on a DeepX-exported YOLO11l OBB model using the yolo CLI.
#
# Usage:
#   bash predict_dxnn_deepx.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}/lib/ultralytics:${PYTHONPATH}"

MODEL_DIR="${SCRIPT_DIR}/yolo11l-obb_deepx_model"
SOURCE="${PROJECT_ROOT}/assets/obb-images"
OUTPUT_DIR="${SCRIPT_DIR}/runs/predict/dxnn/deepx_cli"

yolo predict \
    model="${MODEL_DIR}" \
    source="${SOURCE}" \
    save=True \
    project="${SCRIPT_DIR}" \
    name="${OUTPUT_DIR}" \
    imgsz=640
