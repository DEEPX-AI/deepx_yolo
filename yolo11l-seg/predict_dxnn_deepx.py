"""
YOLO11l-seg DeepX (.dxnn) Predict - Using Ultralytics DeepX Library

Runs inference on a DeepX-exported YOLO11l segmentation model (.dxnn)
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
MODEL_NAME = 'yolo11l-seg'
MODEL_DIR = CURRENT_DIR / f'{MODEL_NAME}_deepx_model'
SOURCE_PATH = PROJECT_ROOT / 'assets' / 'images' / 'bus.jpg'
OUTPUT_DIR = CURRENT_DIR / 'runs' / 'predict' / 'dxnn' / 'deepx'
INPUT_SIZE = 640


def main():
    try:
        # Load the exported DeepX model
        model = YOLO(model=str(MODEL_DIR), task='segment')

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
            print(f"Detected {len(r.boxes)} objects")
            if r.boxes is not None and len(r.boxes) > 0:
                for i, box in enumerate(r.boxes):
                    cls_name = r.names[int(box.cls)]
                    conf = box.conf.item()
                    xyxy = box.xyxy[0].cpu().numpy()
                    print(f"  {i+1}. {cls_name}: {conf:.2f} - [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
            if r.masks is not None:
                print(f"  Masks shape: {r.masks.data.shape}")

        print(f"\nResults saved to '{OUTPUT_DIR}'")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
