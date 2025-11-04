import os
import ssl
import urllib3
from pathlib import Path
from ultralytics import YOLO

def main():
    try:
        # Get the directory where this script is located
        current_dir = Path(__file__).parent
        project_root = current_dir.parent

        model_name = 'yolo11l'
        model_dir = project_root / model_name / 'models'
        model_path = model_dir / f'{model_name}.pt'
        
        # Disable SSL verification (to resolve SSL issues due to corporate firewalls, etc.)
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Set offline mode
        os.environ['YOLO_OFFLINE'] = '1'
        os.environ['ULTRALYTICS_OFFLINE'] = '1'

        # Load torch model
        model = YOLO(model=model_path)

        # Export arguments
        export_args = {
            'opset': 21,
            'dynamic': True,
            'half': False,
            'simplify': True,
            'nms': False
        }

        # Convert model to ONNX format with opset 21
        onnx_model_path = model.export(format='onnx', **export_args)

        print(f"Successfully converted YOLO11l model to ONNX format (opset 21).")
        print(f"{onnx_model_path}' file has been created.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
