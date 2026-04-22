"""
YOLO11l-obb DeepX (.dxnn) Predict - Using Ultralytics DeepX Library

Runs inference on a DeepX-exported YOLO11l oriented bounding box model (.dxnn)
using the custom Ultralytics library with DeepX backend support.

Usage:
    python predict_dxnn_deepx.py
"""
import ultralytics_deepx_lib_setup  # noqa: F401 - must be imported before ultralytics

from pathlib import Path
from ultralytics import YOLO

# Configuration
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_NAME = 'yolo11l-obb'
MODEL_DIR = CURRENT_DIR / f'{MODEL_NAME}_deepx_model'
SOURCE_PATH = PROJECT_ROOT / 'assets' / 'obb-images'
OUTPUT_DIR = CURRENT_DIR / 'runs' / 'predict' / 'dxnn' / 'deepx'
INPUT_SIZE = 640


def main():
    try:
        # Load the exported DeepX model
        model = YOLO(model=str(MODEL_DIR), task='obb')

        # Run inference
        results = model(
            source=str(SOURCE_PATH),
            save=True,
            project=str(CURRENT_DIR),
            name=str(OUTPUT_DIR),
            imgsz=INPUT_SIZE,
        )

        # Process results
        for r in results:
            print(f"Detected {len(r.obb)} objects")
            if r.obb is not None and len(r.obb) > 0:
                for i, obb in enumerate(r.obb):
                    cls_name = r.names[int(obb.cls)]
                    conf = obb.conf.item()
                    print(f"  {i+1}. {cls_name}: {conf:.2f}")

        print(f"\nResults saved to '{OUTPUT_DIR}'")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
