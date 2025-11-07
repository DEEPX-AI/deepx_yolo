"""
YOLOv11 Pose Estimation ONNX Inference using Custom Ultralytics DEEPX Library

This implementation uses the custom Ultralytics DEEPX library for end-to-end inference.
All preprocessing, inference, and postprocessing are handled internally by Ultralytics.

Implementation details:
- Preprocessing: Ultralytics internal letterbox and normalization
- Inference: ONNX Runtime execution via Ultralytics AutoBackend
- Postprocessing: Ultralytics internal NMS and Results generation

Dependencies: cv2, numpy, ultralytics (customized by DEEPX)

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
   └─ [MEASURED] Postprocessing (NMS,keypoint generation, coordinate scaling, Results creation)
   
   What's INCLUDED:
   - Image preprocessing (resize, padding, normalization)
   - Tensor format conversion (numpy ↔ torch, CPU ↔ GPU)
   - model inference execution
   - Non-Maximum Suppression (NMS)
   - Keypoint generation
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
     * predict_onnx_standalone.py: Direct ONNX Runtime without Ultralytics
       → 'runs/predict/onnx/standalone/debug/raw_output/raw_output_[TIMESTAMP].npy'
     * predict_onnx_ultralytics_deepx.py: ONNX model inference with custom library
       → 'runs/predict/onnx/ultralytics_deepx/debug/raw_output/raw_output_[TIMESTAMP].npy'
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
MODEL_EXTENSION = 'onnx'
MODEL_NAME = f'{CURRENT_DIR.name}'
MODEL_FILE = f'{CURRENT_DIR.name}.{MODEL_EXTENSION}'
MODEL_PATH = PROJECT_ROOT / MODEL_NAME / 'models' / MODEL_FILE
SOURCE_PATH = PROJECT_ROOT / 'assets' / 'images' / 'bus.jpg'      # for image file
# SOURCE_PATH = PROJECT_ROOT / 'assets' / 'images'                    # for image directory
OUTPUT_SUBDIR = CURRENT_DIR / 'runs' / 'predict' / MODEL_EXTENSION / "ultralytics_deepx"
DEBUG_OUTPUT_DIR = OUTPUT_SUBDIR / 'debug'   # Directory to save debug outputs
DEBUG_ORIGIN_OUTPUT_DIR = DEBUG_OUTPUT_DIR / 'origin_output'
OUTPUT_DIR = OUTPUT_SUBDIR  # Directory to save results

# Detection parameters (Ultralytics defaults)
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

INPUT_SIZE = 640

# Letterbox preprocessing mode
# False: Square padding (e.g., 640x640) - matches Ultralytics rect=False
# True: Rectangular, preserve aspect ratio (e.g., 480x640) - matches Ultralytics rect=True
# NOTE: ONNX Runtime supports dynamic shapes, so rect=True works fine
# RECT_OPT = True

# NOTE: For DXNN and tolerance calculations, dynamic shapes cannot be used. Set the letterbox rect option to false to match the NPU's fixed shape output.
RECT_OPT = False

# COCO Pose keypoint names (17 keypoints)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO Pose skeleton connections (matching ultralytics)
POSE_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7), (6, 8), (7, 9), (8, 10),
    (9, 11), (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)
]

# COCO Pose colors (matching ultralytics)
POSE_PALETTE = [
    [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
    [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
    [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
]

# Keypoint and limb colors (matching ultralytics)
KPT_COLOR = [16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]
LIMB_COLOR = [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]


def draw_pose_detections(source_path, result, output_path, save=True, show=True):
    """
    Draw pose estimation results on image using Results object.
    Custom visualization implementation using OpenCV.
    """
    image = cv2.imread(source_path)
    
    if result.keypoints is not None:
        # Get pose keypoints data
        keypoints = result.keypoints.xy.cpu().numpy()  # Shape: [num_persons, 17, 2]
        keypoints_conf = result.keypoints.conf.cpu().numpy()  # Shape: [num_persons, 17]
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf.cpu().numpy()  # confidence score of each box
        
        # Line width calculation
        line_width = max(round(sum(image.shape) / 2 * 0.003), 2)
        
        # Draw pose for each person
        for i in range(len(keypoints)):
            box = boxes[i].astype(int)
            conf = confs[i]
            name = names[i]
            person_keypoints = keypoints[i]  # Shape: [17, 2]
            person_conf = keypoints_conf[i]  # Shape: [17]
            
            # Draw bounding box
            box_color = (255, 144, 30)  # Ultralytics default box color (BGR)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), box_color, line_width)
            
            # Draw label with background
            label = f"{name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, max(line_width - 1, 1))[0]
            cv2.rectangle(image, (box[0], box[1] - label_size[1] - 10), 
                         (box[0] + label_size[0], box[1]), box_color, -1)
            cv2.putText(image, label, (box[0], box[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), max(line_width - 1, 1))
            
            # Draw keypoints
            radius = max(line_width - 1, 1)
            conf_thres = 0.25
            
            for j in range(17):  # 17 keypoints
                if person_conf[j] > conf_thres:
                    kx, ky = person_keypoints[j]
                    if kx > 0 and ky > 0:  # Valid keypoint
                        # Get keypoint color
                        color_idx = KPT_COLOR[j]
                        color = POSE_PALETTE[color_idx]
                        # Convert RGB to BGR for OpenCV
                        color_bgr = (color[2], color[1], color[0])
                        cv2.circle(image, (int(kx), int(ky)), radius, color_bgr, -1, lineType=cv2.LINE_AA)
            
            # Draw skeleton connections
            for j, (start_idx, end_idx) in enumerate(POSE_SKELETON):
                # Adjust indices (ultralytics uses 1-based, we use 0-based)
                start_idx_adj = start_idx - 1
                end_idx_adj = end_idx - 1
                
                if 0 <= start_idx_adj < 17 and 0 <= end_idx_adj < 17:
                    start_conf = person_conf[start_idx_adj]
                    end_conf = person_conf[end_idx_adj]
                    start_kpt = person_keypoints[start_idx_adj]
                    end_kpt = person_keypoints[end_idx_adj]
                    
                    # Check confidence and validity
                    if (start_conf > conf_thres and end_conf > conf_thres and
                        start_kpt[0] > 0 and start_kpt[1] > 0 and
                        end_kpt[0] > 0 and end_kpt[1] > 0):
                        
                        # Get limb color
                        limb_color_idx = LIMB_COLOR[j]
                        limb_color_rgb = POSE_PALETTE[limb_color_idx]
                        # Convert RGB to BGR for OpenCV
                        limb_color_bgr = (limb_color_rgb[2], limb_color_rgb[1], limb_color_rgb[0])
                        
                        start_point = (int(start_kpt[0]), int(start_kpt[1]))
                        end_point = (int(end_kpt[0]), int(end_kpt[1]))
                        
                        cv2.line(image, start_point, end_point, limb_color_bgr, 
                               max(int(line_width / 2), 1), lineType=cv2.LINE_AA)
    
    elif result.boxes is not None:
        # Fallback to bounding boxes only if no keypoints
        boxes = result.boxes.xyxy.cpu().numpy()
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        confs = result.boxes.conf.cpu().numpy()
        
        for i in range(len(boxes)):
            box = boxes[i].astype(int)
            conf = confs[i]
            name = names[i]
            # Draw rectangle
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)
            # Draw label
            label = f"{name}: {conf:.2f}"
            cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if save:
        # Save result image
        cv2.imwrite(output_path, image)
        print(f"Pose estimation result saved to '{output_path}' file.")

    if show:
        # Show the image
        cv2.imshow("Pose Estimation Results", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def analyze_results(result, filename):
    """
    Analyze and print pose estimation results from Results object.
    Provides detailed statistics on detected objects.

    Args:
        result (Results): Results object containing boxes and keypoints
        filename (str): Name of the processed file
    """
    
    if result.keypoints is not None and len(result.keypoints) > 0:
        print("="*50)
        print(f"Total pose instances: {len(result.keypoints)}")
        print(f"Keypoints tensor shape: {result.keypoints.data.shape}")
        print(f"Boxes tensor shape: {result.boxes.data.shape}")
        
        # Get confidence values
        confidences = result.boxes.conf.cpu().numpy()
        print(f"Confidence range: {np.min(confidences):.3f} ~ {np.max(confidences):.3f}")
        print(f"Confidences >= 0.25: {np.sum(confidences >= 0.25)}")
        
        # Check class distribution (should be all persons for pose)
        classes = result.boxes.cls.cpu().numpy()
        unique_classes, counts = np.unique(classes, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes.astype(int), counts))}")
        
        # More detailed conf analysis
        conf_bins = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for i in range(len(conf_bins)-1):
            count = np.sum((confidences >= conf_bins[i]) & (confidences < conf_bins[i+1]))
            print(f"Conf {conf_bins[i]:.1f}~{conf_bins[i+1]:.1f}: {count}")
        print(f"Conf >= 0.9: {np.sum(confidences >= 0.9)}")
        
        # Keypoint statistics
        keypoints_conf = result.keypoints.conf.cpu().numpy()  # Shape: [num_persons, 17]
        
        boxes = result.boxes.xyxy.cpu().numpy()
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        
        print(f"[{filename}] Total {len(result.keypoints)} pose instances detected.")
        for i, person_kpt_conf in enumerate(keypoints_conf):
            visible_keypoints = np.sum(person_kpt_conf > 0.5)
            box = boxes[i]
            x1, y1, x2, y2 = box
            print(f"  {i+1}. {names[i]}: {confidences[i]:.2f} - Position: ({x1:.0f}, {y1:.0f}) ~ ({x2:.0f}, {y2:.0f}) - Keypoints: {visible_keypoints}/17 visible")
            
            # Show keypoint details for first person only to avoid too much output
            if i == 0:
                for j, kpt_name in enumerate(KEYPOINT_NAMES):
                    if person_kpt_conf[j] > 0.25:
                        print(f"    {kpt_name}: {person_kpt_conf[j]:.2f}")
        
        print(f"Average visible keypoints per person: {np.mean([np.sum(conf > 0.5) for conf in keypoints_conf]):.1f}/17")
        
    elif result.boxes is not None and len(result.boxes) > 0:
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
        
        print(f"[{filename}] No pose keypoints detected, showing {len(result.boxes)} bounding boxes only.")
        
    else:
        print(f"[{filename}] No pose keypoints or object detections found.")

def run_inference(model_path, image_path, output_dir, debug=False, save=True, show=False, rect=True):
    """
    Run complete inference using specified backend.
    
    Args:
        model_path: Path to model file
        image_path: Path to input image
        output_dir: Directory to save output
        debug: Enable debug mode (saves intermediate outputs)
        save: Save output image with pose estimation
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

        # Load the ONNX model (use task='pose' for pose estimation)
        model = YOLO(model=model_path, task='pose')

        # Debug: Verify model class names
        if debug:
            print("ONNX Model names:", model.names)

        # ============================================================================
        # INFERENCE TIME MEASUREMENT START
        # ============================================================================
        # This measures ONLY the model inference pipeline:
        # 1. Preprocessing (letterbox, normalization, format conversion)
        # 2. Runtime inference execution
        # 3. Postprocessing (NMS, keypoint generation, coordinate scaling, Results creation)
        # ============================================================================
        inference_start = time.perf_counter()
        # NOTE: ONNX Runtime supports dynamic shapes, so rect=True works fine
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

        # Draw pose estimation
        draw_pose_detections(image_path, result, output_path, save=save, show=show)

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
        # NOTE: ONNX Runtime supports dynamic shapes, so rect=True works fine
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
            # NOTE: ONNX Runtime supports dynamic shapes, so rect=True works fine
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
        print(f"  │  └─ Postprocessing:    (NMS, keypoint generation, coordinate scaling)")
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