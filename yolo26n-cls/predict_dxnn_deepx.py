"""
YOLO26n-cls DeepX (.dxnn) Predict - Using Ultralytics DeepX Library

Runs inference on a DeepX-exported YOLO26n classification model (.dxnn)
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
MODEL_NAME = 'yolo26n-cls'
MODEL_DIR = CURRENT_DIR / f'{MODEL_NAME}_deepx_model'
SOURCE_PATH = PROJECT_ROOT / 'assets' / 'images' / 'bus.jpg'
OUTPUT_DIR = CURRENT_DIR / 'runs' / 'predict' / 'dxnn' / 'deepx'
INPUT_SIZE = 224


def main():
    try:
        # Load the exported DeepX model
        model = YOLO(model=str(MODEL_DIR), task='classify')

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
            if r.probs is not None:
                top5_indices = r.probs.top5
                top5_confs = r.probs.top5conf.cpu().numpy()
                print("Top-5 predictions:")
                for idx, conf in zip(top5_indices, top5_confs):
                    cls_name = r.names[idx]
                    print(f"  {cls_name}: {conf:.4f}")

        print(f"\nResults saved to '{OUTPUT_DIR}'")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
