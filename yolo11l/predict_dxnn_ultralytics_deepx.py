"""
YOLOv11 DXNN Inference using Custom Ultralytics DEEPX Library

This implementation uses the custom Ultralytics DEEPX library for end-to-end inference.
All preprocessing, inference, and postprocessing are handled internally by Ultralytics.

Implementation details:
- Preprocessing: Ultralytics internal letterbox and normalization
- Inference: DXNN Runtime execution via Ultralytics AutoBackend
- Postprocessing: Ultralytics internal NMS and Results generation

Dependencies: cv2, numpy, dx_engine, ultralytics (customized by DEEPX)

===================================================================================
PERFORMANCE MEASUREMENT BREAKDOWN
===================================================================================

This script measures two types of performance metrics:

1. INFERENCE TIME (Pure Inference Only)
   -----------------------------------------------------------------------
   Measures ONLY the time spent in: model(source=image_path, ...)
   
   Pipeline Scope:
   ├─ [MEASURED] Preprocessing (letterbox, normalization, HWC→CHW conversion)
   ├─ [MEASURED] Runtime Inference (model execution on NPU/GPU)
   └─ [MEASURED] Postprocessing (NMS, coordinate scaling, Results creation)
   
   What's INCLUDED:
   - Image preprocessing (resize, padding, normalization)
   - Tensor format conversion (numpy ↔ torch, CPU ↔ GPU)
   - model inference execution
   - Non-Maximum Suppression (NMS)
   - Bounding box coordinate transformation
   - Results object creation
   
   What's EXCLUDED:
   - Image file I/O (imread/imwrite)
   - Result visualization (drawing boxes on images)
   - Progress printing and statistics calculation

2. OVERALL PROCESSING TIME (Overall Pipeline)
   -----------------------------------------------------------------------
   Measures the ENTIRE end-to-end processing time
   
   Pipeline Scope:
   ├─ Image file loading
   ├─ [INFERENCE TIME] <- All inference components (see above)
   ├─ Result visualization (drawing boxes/labels)
   ├─ Image file saving
   └─ Statistics calculation and display

===================================================================================
"""

# IMPORTANT: Import this BEFORE any ultralytics imports
import ultralytics_deepx_lib_setup
"""
'ultralytics_deepx_lib_setup' imports the custom Ultralytics DEEPX library from 'lib/ultralytics'
(defined in '.gitmodules').

The custom Ultralytics DEEPX library includes modifications to 'lib/ultralytics/ultralytics/nn/autobackend.py'
that enable the following debugging and DXNN features:

1. DXNN model (.dxnn) inference support
   - Enables DEEPX runtime for DXNN model inference
   - See 'predict_dxnn_ultralytics_deepx.py' run_inference() for usage example

2. Debug: model input tensor visualization and saving
   - Saves preprocessed input tensors to:
     'runs/predict/[MODEL_EXTENSION]/ultralytics_deepx/debug/input/preprocessed_input_[TIMESTAMP].jpg'

3. Debug: model raw output tensor saving and comparison
   - Saves raw output tensors to:
     'runs/predict/[MODEL_EXTENSION]/ultralytics_deepx/debug/raw_output/raw_output_[TIMESTAMP].npy'
   - Use 'util/compare_raw_outputs.py' to compare outputs with other implementations:
     * predict_dxnn_standalone.py: Direct DXNN Runtime without Ultralytics
       → 'runs/predict/dxnn/standalone/debug/raw_output/raw_output_[TIMESTAMP].npy'
     * predict_dxnn_ultralytics_deepx.py: DXNN model inference with custom library
       → 'runs/predict/dxnn/ultralytics_deepx/debug/raw_output/raw_output_[TIMESTAMP].npy'
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

# Configuration
DEBUG_MODE = 1  # Set to 1 to enable AutoBackend debug output, 0 to disable

# Set environment variable for AutoBackend to check
import os
os.environ['DEEPX_DEBUG_MODE'] = str(DEBUG_MODE)

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_EXTENSION = 'dxnn'
MODEL_NAME = f'{CURRENT_DIR.name}'
MODEL_FILE = f'{CURRENT_DIR.name}.{MODEL_EXTENSION}'
MODEL_PATH = PROJECT_ROOT / MODEL_NAME / 'models' / MODEL_FILE
# SOURCE_PATH = PROJECT_ROOT / 'assets' / 'images' / 'bus.jpg'      # for image file
SOURCE_PATH = PROJECT_ROOT / 'assets' / 'images'                    # for image directory
OUTPUT_SUBDIR = CURRENT_DIR / 'runs' / 'predict' / MODEL_EXTENSION / "ultralytics_deepx"
DEBUG_OUTPUT_DIR = OUTPUT_SUBDIR / 'debug'   # Directory to save debug outputs
DEBUG_ORIGIN_OUTPUT_DIR = DEBUG_OUTPUT_DIR / 'origin_output'
OUTPUT_DIR = OUTPUT_SUBDIR  # Directory to save results

# COCO class names (based on dataset that YOLOv11 was trained on)
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def draw_detections(source_path, result, output_path, save=True, show=True):
    """
    Draw bounding boxes on image using Ultralytics Results object.
    Custom visualization implementation using OpenCV.
    """
    image = cv2.imread(source_path)
    
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy  # x1, y1, x2, y2 format
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box

        # Color palette (different colors for each class)
        colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        
        for i in range(len(xyxy)):
            box = xyxy[i].cpu().numpy().astype(int)
            conf = confs[i].item()
            name = names[i]
            class_id = result.boxes.cls[i].item()
            
            x1, y1, x2, y2 = box
            
            # Get class-specific color
            color = colors[int(class_id)].tolist()
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if save:
        # Save result image
        cv2.imwrite(output_path, image)
        print(f"Detection result saved to '{output_path}' file.")

    if show:
        # Show the image
        cv2.imshow("Object Detections", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def analyze_results(result, filename):
    """
    Analyze and print detection results from Ultralytics Results object.
    Provides detailed statistics on detected objects.

    Args:
        result (Results): Results object containing boxes and masks
        filename (str): Name of the processed file
    """
    if result.boxes is None or len(result.boxes) == 0:
        print(f"[{filename}] No objects detected.")
        return

    print("="*50)
    print(f"Total object detections: {len(result.boxes)}")
    print(f"Boxes tensor shape: {result.boxes.data.shape}")
    
    # Get confidence values
    confidences = result.boxes.conf.cpu().numpy()
    print(f"Confidence range: {np.min(confidences):.3f} ~ {np.max(confidences):.3f}")
    print(f"Confidences >= 0.25: {np.sum(confidences >= 0.25)}")

    # Check class distribution
    classes = result.boxes.cls.cpu().numpy()
    unique_classes, counts = np.unique(classes, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_classes.astype(int), counts))}")

    # More detailed conf analysis
    conf_bins = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(len(conf_bins)-1):
        count = np.sum((confidences >= conf_bins[i]) & (confidences < conf_bins[i+1]))
        print(f"Conf {conf_bins[i]:.1f}~{conf_bins[i+1]:.1f}: {count}")
    print(f"Conf >= 0.9: {np.sum(confidences >= 0.9)}")
    
    # Detailed detection info
    xyxy = result.boxes.xyxy
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
    confs = result.boxes.conf
    
    print(f"[{filename}] Total {len(result.boxes)} objects detected.")
    for i in range(len(result.boxes)):
        class_name = names[i]
        score = confs[i].item()
        box = xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = box
        
        print(f"  {i+1}. {class_name}: {score:.2f} - Position: ({x1:.0f}, {y1:.0f}) ~ ({x2:.0f}, {y2:.0f})")

def run_inference(model_path, image_path, output_dir, debug=False, save=True, show=False):
    """
    Run complete inference using specified backend.
    
    Args:
        model_path: Path to model file
        image_path: Path to input image
        output_dir: Directory to save output
        debug: Enable debug mode (saves intermediate outputs)
        save: Save output image with detections
        show: Display output image
    
    Returns:
        tuple: (output_path, statistics_dict) if successful, (None, None) otherwise
        statistics_dict contains: inference_time, total_time
    """
    try:
        import time
        
        # ============================================================================
        # TOTAL PROCESSING TIME MEASUREMENT START (Overall Pipeline)
        # ============================================================================
        start_time = time.perf_counter()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

        # Process image
        print(f"\nProcessing: {Path(image_path).name}")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if debug:
            print("[INFO] Debug mode enabled, Ultralytics DEEPX saves debug data(preprocessed image, raw output).")

        # Load the ONNX model (use task='detect' for object detection)
        model = YOLO(model=model_path, task='detect')

        # Debug: Verify model class names
        print("DXNN Model names:", model.names)

        # ============================================================================
        # INFERENCE TIME MEASUREMENT START
        # ============================================================================
        # This measures ONLY the model inference pipeline:
        # 1. Preprocessing (letterbox, normalization, format conversion)
        # 2. Runtime inference execution
        # 3. Postprocessing (NMS, coordinate scaling, Results creation)
        # ============================================================================
        inference_start = time.perf_counter()
        results = model(source=image_path, save=save, project=CURRENT_DIR, name=DEBUG_ORIGIN_OUTPUT_DIR)
        inference_time = time.perf_counter() - inference_start
        # ============================================================================
        # INFERENCE TIME MEASUREMENT END
        # ============================================================================
        
        result = results[0]  # Get first (and only) result

        # 4. Visualization and analysis (NOT included in inference_time)
        filename = Path(image_path).stem
        output_filename = Path(image_path).stem + f'_detected_{timestamp}.jpg'
        output_path = str(Path(output_dir) / output_filename)

        # Draw detections
        draw_detections(image_path, result, output_path, save=save, show=show)

        # Print analysis result
        analyze_results(result, filename)

        # ============================================================================
        # TOTAL PROCESSING TIME MEASUREMENT END
        # ============================================================================
        overall_time = time.perf_counter() - start_time
        
        # Print timing statistics
        print(f"\n[Timing] Inference: {inference_time:.3f}s | Total: {overall_time:.3f}s")
        
        # Prepare statistics dictionary
        stats = {
            'inference_time': inference_time,
            'overall_time': overall_time,
            'filename': Path(image_path).name
        }
        
        return output_path, stats

    except Exception as e:
        print(f"[{Path(image_path).name}] Error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """
    Main function to process single image or directory of images.
    Supports batch processing and provides detailed summary.
    """
    saved_files = []
    image_statistics = {}  # Store statistics for each image
    
    source_path = Path(SOURCE_PATH)

    # Check if source is file or directory
    if source_path.is_file():
        print("Processing single image file.")
        print(f"Results will be saved in '{OUTPUT_DIR}' folder.")
        print("-" * 50)

        result_path, stats = run_inference(MODEL_PATH, str(source_path), OUTPUT_DIR, debug=DEBUG_MODE)
        if result_path:
            saved_files.append(result_path)
            image_statistics[Path(result_path).name] = stats

    elif source_path.is_dir():
        print("Processing directory of images.")
        print(f"Results will be saved in '{OUTPUT_DIR}' folder.")
        print("-" * 50)
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(ext))
        
        if not image_files:
            print(f"No image files found in {source_path}")
            return
        
        print(f"Found {len(image_files)} images\n")
        
        # Process each image
        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
            print("-" * 50)
            result_path, stats = run_inference(MODEL_PATH, str(image_path), OUTPUT_DIR, debug=True)
            if result_path:
                saved_files.append(result_path)
                image_statistics[Path(result_path).name] = stats
    else:
        print(f"Error: {source_path} is neither a file nor a directory")
        return

    # Print summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total images processed: {len(saved_files)}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Calculate total statistics
    total_inference_time = sum(stats['inference_time'] for stats in image_statistics.values() if stats)
    total_overall_time = sum(stats['overall_time'] for stats in image_statistics.values() if stats)
    
    if image_statistics:
        print(f"\n{'='*80}")
        print("IMAGE PROCESSING STATISTICS")
        print(f"{'='*80}")
        
        for idx, file_path in enumerate(saved_files, 1):
            filename = Path(file_path).name
            
            print(f"\n[{idx}] {filename}")
            
            if filename in image_statistics:
                stats = image_statistics[filename]
                if stats:
                    inference_pct = (stats['inference_time'] / stats['overall_time'] * 100) if stats['overall_time'] > 0 else 0
                    fps_overall = (1.0 / stats['overall_time']) if stats['overall_time'] > 0 else 0
                    fps_inference = (1.0 / stats['inference_time']) if stats['inference_time'] > 0 else 0
                    print(f"    Source:                {stats['filename']}")
                    print(f"    Total Processing Time: {stats['overall_time']:.3f}s")
                    print(f"    Pure Inference Time:   {stats['inference_time']:.3f}s ({inference_pct:.1f}%)")
                    print(f"    FPS (Overall):         {fps_overall:.2f}")
                    print(f"    FPS (Inference Only):  {fps_inference:.2f}")
        
        # Aggregate statistics
        num_images = len(image_statistics)
        inference_percentage = (total_inference_time / total_overall_time * 100) if total_overall_time > 0 else 0
        avg_inference_time = total_inference_time / num_images if num_images > 0 else 0
        avg_overall_time = total_overall_time / num_images if num_images > 0 else 0
        
        print(f"\n{'='*80}")
        print("AGGREGATE STATISTICS (All Images)")
        print(f"{'='*80}")
        print(f"Total Images:                       {num_images}")
        print(f"Total Overall Processing Time:      {total_overall_time:.3f}s")
        print(f"Total Pure Inference Time:          {total_inference_time:.3f}s ({inference_percentage:.1f}%)")
        print(f"Average Overall Time:                 {avg_overall_time:.3f}s per image")
        print(f"Average Pure Inference Time:             {avg_inference_time:.3f}s per image")

        # Calculate aggregate FPS
        total_fps_overall = (num_images / total_overall_time) if total_overall_time > 0 else 0
        total_fps_inference = (num_images / total_inference_time) if total_inference_time > 0 else 0
        print(f"Total FPS (Overall):                {total_fps_overall:.2f}")
        print(f"Total FPS (Inference Only):         {total_fps_inference:.2f}")
        
        # Performance breakdown explanation
        print(f"\n{'='*80}")
        print("PERFORMANCE BREAKDOWN EXPLANATION")
        print(f"{'='*80}")
        overhead_time = total_overall_time - total_inference_time
        overhead_percentage = (overhead_time / total_overall_time * 100) if total_overall_time > 0 else 0
        print(f"\nTotal Processing Time Breakdown:")
        print(f"  ├─ Pure Inference Time:  {total_inference_time:.2f}s ({inference_percentage:.1f}%)")
        print(f"  │  ├─ Preprocessing:     (letterbox, normalization)")
        print(f"  │  ├─ Inference:         (NPU/GPU execution)")
        print(f"  │  └─ Postprocessing:    (NMS, coordinate scaling)")
        print(f"  │")
        print(f"  └─ Overhead Time:        {overhead_time:.2f}s ({overhead_percentage:.1f}%)")
        print(f"     ├─ I/O:               (imread, imwrite)")
        print(f"     ├─ Visualization:     (drawing boxes, labels)")
        print(f"     └─ Misc:              (statistics, printing)")
    
    print(f"\n{'='*80}")
    print("Processing completed successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()