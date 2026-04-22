import os
import ssl
import urllib3
from pathlib import Path
import ultralytics_deepx_lib_setup  # noqa: F401 - must be imported before ultralytics
from ultralytics import YOLO

def main():
    try:
        # Get the directory where this script is located
        current_dir = Path(__file__).parent
        project_root = current_dir.parent

        model_name = 'yolo11l-pose'
        model_dir = project_root / model_name / 'models'
        model_path = model_dir / f'{model_name}.pt'
        
        # Disable SSL verification (to resolve SSL issues due to corporate firewalls, etc.)
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Set offline mode (0 = online, allows dataset download)
        os.environ['YOLO_OFFLINE'] = '0'
        os.environ['ULTRALYTICS_OFFLINE'] = '0'

        # Load torch model
        model = YOLO(model=model_path)

        # Export arguments for DeepX format
        # - int8: INT8 quantization (required for DeepX, automatically enforced)
        # - data: calibration dataset for INT8 quantization
        # - fraction: fraction of calibration dataset to use
        # - imgsz: input image size
        export_args = {
            'int8': True,
            'data': 'coco128.yaml',
            'fraction': 1.0,
            'imgsz': 640,
        }

        # Convert model to DeepX format (.dxnn)
        dxnn_model_path = model.export(format='deepx', **export_args)

        print(f"Successfully converted {model_name} model to DeepX format (.dxnn).")
        print(f"'{dxnn_model_path}' directory has been created.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
