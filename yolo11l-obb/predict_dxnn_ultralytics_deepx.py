"""
YOLOv11 OBB DXNN Inference using Custom Ultralytics DEEPX Library

This implementation uses the custom Ultralytics DEEPX library for end-to-end inference.
All preprocessing, inference, and postprocessing are handled internally by Ultralytics.

Implementation details:
- Preprocessing: Ultralytics internal letterbox and normalization
- Inference: DXNN Runtime execution via Ultralytics AutoBackend
- Postprocessing: Ultralytics internal NMS, Regularization of rotated boxes and Results generation

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
   └─ [MEASURED] Postprocessing (NMS, Regularize rotated boxes, coordinate scaling, Results creation)
   
   What's INCLUDED:
   - Image preprocessing (resize, padding, normalization)
   - Tensor format conversion (numpy ↔ torch, CPU ↔ GPU)
   - model inference execution
   - Non-Maximum Suppression (NMS)
   - Regularization of rotated boxes
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
SOURCE_PATH = PROJECT_ROOT / 'assets' / 'images' / 'boats.jpg'      # for image file
# SOURCE_PATH = PROJECT_ROOT / 'assets' / 'images'                    # for image directory
OUTPUT_SUBDIR = CURRENT_DIR / 'runs' / 'predict' / MODEL_EXTENSION / "ultralytics_deepx"
DEBUG_OUTPUT_DIR = OUTPUT_SUBDIR / 'debug'   # Directory to save debug outputs
DEBUG_ORIGIN_OUTPUT_DIR = DEBUG_OUTPUT_DIR / 'origin_output'
OUTPUT_DIR = OUTPUT_SUBDIR  # Directory to save results

INPUT_SIZE = 1024

# Letterbox preprocessing mode
# False: Square padding (e.g., 640x640) - matches Ultralytics rect=False
# True: Rectangular, preserve aspect ratio (e.g., 480x640) - matches Ultralytics rect=True
# IMPORTANT: rect=False forces square padding (1024x1024) for DEEPX NPU fixed input shape
RECT_OPT = False

# DOTAv1.0 class names (dataset that YOLOv11-obb was trained on)
CLASSES = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court', 
           'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 
           'roundabout', 'soccer-ball-field', 'swimming-pool']

def draw_detections(source_path, result, output_path, save=True, show=True):
    """
    Draw oriented bounding boxes on image using Ultralytics Results object.
    Custom visualization implementation using OpenCV.
    """
    image = cv2.imread(source_path)
    
    if result.obb is not None:
        xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
        xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
        names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
        confs = result.obb.conf  # confidence score of each box

        # Color palette (different colors for each class)
        colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        
        for i in range(len(xywhr)):
            poly = xyxyxyxy[i].cpu().numpy().astype(int)
            conf = confs[i].item()
            name = names[i]
            class_id = result.obb.cls[i].item()
            
            # Get class-specific color
            color = colors[int(class_id)].tolist()
            
            # Draw oriented bounding box (polygon)
            cv2.drawContours(image, [poly.reshape(-1, 2)], 0, color, 3)
            
            # Get center point for drawing
            center_x, center_y = int(xywhr[i][0].item()), int(xywhr[i][1].item())
            cv2.circle(image, (center_x, center_y), 3, color, -1)
            
            # Calculate label position (use min x,y of polygon)
            min_x, min_y = np.min(poly.reshape(-1, 2), axis=0)
            
            # Draw label with background
            label = f"{name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (min_x, min_y - label_size[1] - 10), 
                         (min_x + label_size[0], min_y), color, -1)
            cv2.putText(image, label, (min_x, min_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if save:
        # Save result image
        cv2.imwrite(output_path, image)
        print(f"OBB detection result saved to '{output_path}' file.")

    if show:
        # Show the image
        cv2.imshow("OBB Detections", image)
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
    if result.obb is None or len(result.obb) == 0:
        print(f"[{filename}] No oriented objects detected.")
        return

    print("="*50)
    print(f"Total OBB detections: {len(result.obb)}")
    print(f"OBB tensor shape: {result.obb.data.shape}")
    
    # Get confidence values
    confidences = result.obb.conf.cpu().numpy()
    print(f"Confidence range: {np.min(confidences):.3f} ~ {np.max(confidences):.3f}")
    print(f"Confidences >= 0.25: {np.sum(confidences >= 0.25)}")
    
    # Check class distribution
    classes = result.obb.cls.cpu().numpy()
    unique_classes, counts = np.unique(classes, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_classes.astype(int), counts))}")
    
    # More detailed conf analysis
    conf_bins = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(len(conf_bins)-1):
        count = np.sum((confidences >= conf_bins[i]) & (confidences < conf_bins[i+1]))
        print(f"Conf {conf_bins[i]:.1f}~{conf_bins[i+1]:.1f}: {count}")
    print(f"Conf >= 0.9: {np.sum(confidences >= 0.9)}")
    
    # Detailed detection info
    xywhr = result.obb.xywhr
    names = [result.names[cls.item()] for cls in result.obb.cls.int()]
    confs = result.obb.conf
    
    print(f"[{filename}] Total {len(result.obb)} oriented objects detected.")
    for i in range(len(result.obb)):
        class_name = names[i]
        score = confs[i].item()
        xywhr_data = xywhr[i].cpu().numpy()
        cx, cy, w, h, angle = xywhr_data
        
        # Convert to corner coordinates for display
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        
        print(f"  {i+1}. {class_name}: {score:.2f} - Position: ({x1:.0f}, {y1:.0f}) ~ ({x2:.0f}, {y2:.0f}) - OBB: ({cx:.1f}, {cy:.1f}, {w:.1f}x{h:.1f}, {np.degrees(angle):.1f}°)")

def run_inference(model_path, image_path, output_dir, debug=False, save=True, show=False, rect=True):
    """
    Run complete inference using specified backend.
    
    Args:
        model_path: Path to model file
        image_path: Path to input image
        output_dir: Directory to save output
        debug: Enable debug mode (saves intermediate outputs)
        save: Save output image with detections
        show: Display output image
        rect: Enable rectangular inference (preserve aspect ratio)
              - True: Preserve aspect ratio (e.g., 480x640) - default
              - False: Force square padding (e.g., 640x640)
    
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

        # Load the DXNN model (use task='obb' for oriented object detection)
        model = YOLO(model=model_path, task='obb')

        # Debug: Verify model class names
        if debug:
            print("DXNN Model names:", model.names)

        # ============================================================================
        # INFERENCE TIME MEASUREMENT START
        # ============================================================================
        # This measures ONLY the model inference pipeline:
        # 1. Preprocessing (letterbox, normalization, format conversion)
        # 2. Runtime inference execution
        # 3. Postprocessing (NMS, Regularization of rotated boxes, coordinate scaling, Results creation)
        # ============================================================================
        inference_start = time.perf_counter()
        results = model(source=image_path, save=save, project=CURRENT_DIR, name=DEBUG_ORIGIN_OUTPUT_DIR, imgsz=INPUT_SIZE, rect=rect, verbose=debug)
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
        if debug:
            analyze_results(result, filename)

        # ============================================================================
        # TOTAL PROCESSING TIME MEASUREMENT END
        # ============================================================================
        overall_time = time.perf_counter() - start_time
        
        # Print timing statistics
        if debug:
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
        print(f"Letterbox mode: {'rect (preserve aspect ratio)' if RECT_OPT else 'square padding'}")
        print(f"Results will be saved in '{OUTPUT_DIR}' folder.")
        print("-" * 50)
        # IMPORTANT: rect=False forces square padding (1024x1024) for DEEPX NPU fixed input shape
        result_path, stats = run_inference(MODEL_PATH, str(source_path), OUTPUT_DIR, debug=(DEBUG_MODE == 1), rect=RECT_OPT)
        if result_path:
            saved_files.append(result_path)
            image_statistics[Path(result_path).name] = stats

    elif source_path.is_dir():
        print("Processing directory of images.")
        print(f"Letterbox mode: {'rect (preserve aspect ratio)' if RECT_OPT else 'square padding'}")
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
            # IMPORTANT: rect=False forces square padding (1024x1024) for DEEPX NPU fixed input shape
            result_path, stats = run_inference(MODEL_PATH, str(image_path), OUTPUT_DIR, debug=(DEBUG_MODE == 1), rect=RECT_OPT)
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
        print(f"  │  └─ Postprocessing:    (NMS, Regularization of rotated boxes, coordinate scaling)")
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