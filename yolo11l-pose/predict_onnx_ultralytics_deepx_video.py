"""
YOLOv11 Pose Estimation ONNX Inference using Custom Ultralytics DEEPX Library

This implementation uses the custom Ultralytics DEEPX library for end-to-end inference.
All preprocessing, inference, and postprocessing are handled internally by Ultralytics.
Supports both image and video processing.

Implementation details:
- Preprocessing: Ultralytics internal letterbox and normalization
- Inference: ONNX Runtime execution via Ultralytics AutoBackend
- Postprocessing: Ultralytics internal NMS, Keypoint generation, and Results generation
- Video Processing: Frame-by-frame inference with video output generation

Dependencies: cv2, numpy, ultralytics (customized by DEEPX)

===================================================================================
PERFORMANCE MEASUREMENT BREAKDOWN
===================================================================================

This script measures two types of performance metrics:

1. INFERENCE TIME (Pure Inference Only)
   -----------------------------------------------------------------------
   Measures ONLY the time spent in: model(source=frame_batch, ...)
   
   Pipeline Scope:
   ├─ [MEASURED] Preprocessing (letterbox, normalization, HWC→CHW conversion)
   ├─ [MEASURED] Runtime Inference (run_batch execution on NPU/GPU)
   └─ [MEASURED] Postprocessing (NMS, keypoint generation, coordinate scaling, Results creation)
   
   What's INCLUDED:
   - Image preprocessing (resize, padding, normalization)
   - Tensor format conversion (numpy ↔ torch, CPU ↔ GPU)
   - model inference execution
   - Non-Maximum Suppression (NMS)
   - Keypoint generation
   - Bounding box coordinate transformation
   - Results object creation
   
   What's EXCLUDED:
   - Video I/O (reading frames, writing output)
   - Frame batching/accumulation logic
   - Result visualization (drawing boxes on frames)
   - Progress printing and statistics calculation

2. OVERALL PROCESSING TIME (Overall Pipeline)
   -----------------------------------------------------------------------
   Measures the ENTIRE end-to-end processing time
   
   Pipeline Scope:
   ├─ Video file opening and property reading
   ├─ Frame reading from video (cv2.VideoCapture.read())
   ├─ Frame batching and accumulation
   ├─ [INFERENCE TIME] <- All inference components (see above)
   ├─ Result visualization (result.plot() - drawing boxes/labels)
   ├─ Video writing (cv2.VideoWriter.write())
   ├─ Progress display and statistics calculation
   └─ Video file closing and cleanup
   
   Additional Overhead in Overall Time:
   - cv2.VideoCapture operations (~5-10%)
   - cv2.VideoWriter operations (~10-15%)
   - Frame buffer management
   - result.plot() visualization (~5-10%)
   - Progress printing and timing calculations

TYPICAL TIME DISTRIBUTION:
   - Pure Inference: ~70% of total time
   - Video I/O: ~20% of total time  
   - Visualization & Misc: ~10% of total time

FPS METRICS:
   - FPS (Inference Only) = frames / inference_time
     → Represents pure model throughput (what the model can achieve)
   
   - FPS (Overall) = frames / overall_time
     → Represents real-world application performance (including all overheads)

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
DEBUG_MODE = 0  # Set to 1 to enable AutoBackend debug output, 0 to disable
ASYNC_MODE = 1  # Set to 1 to enable async batch processing, 0 to disable
BATCH_SIZE = 12 if ASYNC_MODE else 1  # if ASYNC_MODE is disabled, use batch size 1

# Set environment variable for AutoBackend to check
import os
os.environ['DEEPX_DEBUG_MODE'] = str(DEBUG_MODE)
os.environ['DEEPX_ASYNC_MODE'] = str(ASYNC_MODE)  # Enable async batch processing

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_EXTENSION = 'onnx'
MODEL_NAME = f'{CURRENT_DIR.name}'
MODEL_FILE = f'{CURRENT_DIR.name}.{MODEL_EXTENSION}'
MODEL_PATH = PROJECT_ROOT / MODEL_NAME / 'models' / MODEL_FILE
# SOURCE_PATH = PROJECT_ROOT / 'assets' / 'images' / 'bus.jpg'              # for image file
# SOURCE_PATH = PROJECT_ROOT / 'assets' / 'images'                          # for image directory
SOURCE_PATH = PROJECT_ROOT / 'assets' / 'videos' / 'dance-group2.mov'       # for video file
# SOURCE_PATH = PROJECT_ROOT / 'assets' / 'videos'                          # for video directory
# SOURCE_PATH = PROJECT_ROOT / 'assets'                                     # for image and video directory
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
RECT_OPT = True

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


def process_frame_batch(model, frame_batch, video_writer, save, show, debug=False, rect=True):
    """
    Process a batch of frames through the model and handle results.
    
    Args:
        model: YOLO model instance
        frame_batch: List of frames to process
        video_writer: VideoWriter object for saving output
        save: Whether to save output video
        show: Whether to display output
        rect: Enable rectangular inference (preserve aspect ratio)
              - True: Preserve aspect ratio (e.g., 480x640) - default
              - False: Force square padding (e.g., 640x640)
        
    Returns:
        tuple: (Number of frames processed, inference time in seconds)
        
    Timing Breakdown:
        inference_time: ONLY measures model(source=...) execution
            - Includes: preprocessing, inference, postprocessing (NMS, keypoint generation)
            - Excludes: visualization (result.plot()), video I/O
    """
    if not frame_batch:
        return 0, 0.0
    
    import time
    
    # ============================================================================
    # INFERENCE TIME MEASUREMENT START
    # ============================================================================
    # This measures ONLY the model inference pipeline:
    # 1. Preprocessing (letterbox, normalization, format conversion)
    # 2. Runtime batch inference execution
    # 3. Postprocessing (NMS, keypoint generation, coordinate scaling, Results creation)
    # ============================================================================
    inference_start = time.perf_counter()
    results = model(source=frame_batch, save=False, project=CURRENT_DIR, name=DEBUG_ORIGIN_OUTPUT_DIR, imgsz=INPUT_SIZE, rect=rect, verbose=debug)
    inference_time = time.perf_counter() - inference_start
    # ============================================================================
    # INFERENCE TIME MEASUREMENT END
    # ============================================================================
    
    processed_count = 0
    # Process each result in the batch (NOT included in inference_time)
    for result in results:
        # Create annotated frame (visualization overhead - NOT in inference_time)
        annotated_frame = result.plot()
        
        # Write frame to output video if saving
        if save and video_writer is not None:
            video_writer.write(annotated_frame)
        
        # Display frame if requested
        if show:
            cv2.imshow("Video Detection Results", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Processing interrupted by user")
                return -1, 0.0  # Signal interruption
        
        processed_count += 1
    
    return processed_count, inference_time


def run_video_inference(model_path, video_path, output_dir, debug=False, save=True, show=False, rect=True):
    """
    Run complete inference on video using specified backend.
    
    Args:
        model_path: Path to model file
        video_path: Path to input video
        output_dir: Directory to save output
        debug: Enable debug mode (saves intermediate outputs)
        save: Save output video with pose estimation
        show: Display output video
        rect: Enable rectangular inference (preserve aspect ratio)
              - True: Preserve aspect ratio (e.g., 480x640) - default
              - False: Force square padding (e.g., 640x640)
    
    Returns:
        tuple: (output_path, statistics_dict) if successful, (None, None) otherwise
        statistics_dict contains: batch_size, frame_count, overall_time, inference_time, fps
        
    Timing Measurements:
        overall_time (Overall Pipeline):
            - Measured from function start to end
            - Includes: video I/O, frame batching, inference, visualization, progress display
            
        inference_time (Pure Inference):
            - Sum of all batch inference times from process_frame_batch()
            - Includes ONLY: preprocessing + inference + postprocessing (NMS, keypoint generation)
            - Excludes: video I/O, visualization, batching logic
    """
    try:
        import time
        # ============================================================================
        # OVERALL PROCESSING TIME MEASUREMENT START (Overall Pipeline)
        # ============================================================================
        # This measures the ENTIRE end-to-end pipeline including:
        # - Video file opening and property reading
        # - Frame reading loop (cv2.VideoCapture.read())
        # - Frame batching and accumulation
        # - Inference (preprocessing + inferencing + postprocessing)
        # - Result visualization (result.plot())
        # - Video writing (cv2.VideoWriter.write())
        # - Progress display and statistics
        # - Video file closing and cleanup
        # ============================================================================
        start_time = time.perf_counter()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        
        print(f"\nProcessing video: {Path(video_path).name}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if debug:
            print("[INFO] Debug mode enabled, Ultralytics DEEPX saves debug data(preprocessed image, raw output).")

        # Load the ONNX model (use task='pose' for pose estimation)
        model = YOLO(model=model_path, task='pose')
        
        # Debug: Verify model class names
        if debug:
            print("ONNX Model names:", model.names)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if saving
        video_writer = None
        if save:
            video_filename = Path(video_path).stem + f'_detected_{timestamp}.mp4'
            output_video_path = str(Path(output_dir) / video_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"Output video will be saved to: {output_video_path}")
        
        frame_count = 0
        processed_frames = 0
        frame_batch = []
        total_inference_time = 0.0  # Track pure inference time
        
        print(f"Starting video processing with batch size: {BATCH_SIZE}")
        print("-" * 50)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                # Process remaining frames in batch if any
                if frame_batch:
                    count, inf_time = process_frame_batch(model, frame_batch, video_writer, save, show, rect=rect, debug=debug)
                    if count == -1:  # User interrupted
                        break
                    processed_frames += count
                    total_inference_time += inf_time
                    frame_batch.clear()
                break
                
            frame_count += 1
            frame_batch.append(frame)
            
            # Process batch when it reaches BATCH_SIZE
            if len(frame_batch) == BATCH_SIZE:
                # Run batch inference using Ultralytics YOLO class
                # The YOLO class internally handles:
                # 1. Preprocessing: letterbox, normalization, channel conversion
                # 2. Inference: Runtime execution via AutoBackend (batch processing)
                # 3. Postprocessing: NMS, keypoint generation, coordinate scaling, Results object creation
                count, inf_time = process_frame_batch(model, frame_batch, video_writer, save, show, rect=rect, debug=debug)
                if count == -1:  # User interrupted
                    break
                processed_frames += count
                total_inference_time += inf_time
                
                # Calculate batch statistics
                progress = (frame_count / total_frames) * 100
                batch_fps = count / inf_time if inf_time > 0 else 0
                cumulative_fps = processed_frames / total_inference_time if total_inference_time > 0 else 0
                
                # Show progress with detailed statistics
                print(f"Processing frames {frame_count - BATCH_SIZE + 1}-{frame_count}/{total_frames} ({progress:.1f}%) | "
                      f"Batch: {inf_time:.2f}s, {batch_fps:.2f} FPS | "
                      f"Cumulative: {cumulative_fps:.2f} FPS")
                
                # Clear batch for next iteration
                frame_batch.clear()
        
        # Cleanup
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if show:
            cv2.destroyAllWindows()
        
        # ============================================================================
        # OVERALL PROCESSING TIME MEASUREMENT END (Overall Pipeline)
        # ============================================================================
        overall_time = time.perf_counter() - start_time
        # ============================================================================
        
        # Calculate performance statistics
        fps_overall = processed_frames / overall_time if overall_time > 0 else 0.0
        fps_inference = processed_frames / total_inference_time if total_inference_time > 0 else 0.0
        
        print(f"\nVideo processing completed!")
        print(f"Processed {processed_frames}/{total_frames} frames")
        print(f"Total time: {overall_time:.2f}s, Pure inference time: {total_inference_time:.2f}s")
        print(f"FPS (Overall): {fps_overall:.2f}, FPS (Inference Only): {fps_inference:.2f}")
        
        # Prepare statistics dictionary
        stats = {
            'batch_size': BATCH_SIZE,
            'frame_count': processed_frames,
            'overall_time': overall_time,
            'inference_time': total_inference_time,
            'fps': fps
        }
        
        if save and video_writer is not None:
            print(f"Output video saved to: {output_video_path}")
            return output_video_path, stats
        
        return None, stats
        
    except Exception as e:
        print(f"[{Path(video_path).name}] Error occurred during video processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

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

def is_video_file(filepath):
    """
    Check if the file is a video file based on its extension.
    
    Args:
        filepath: Path to the file
        
    Returns:
        bool: True if it's a video file, False otherwise
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.3gp', '.webm']
    return Path(filepath).suffix.lower() in video_extensions


def get_media_files(source_path):
    """
    Get list of media files from source path (file or directory).
    Recursively searches subdirectories including symbolic links.
    
    Args:
        source_path: Path to file or directory
        
    Returns:
        list: List of Path objects for media files
    """
    source_path = Path(source_path)
    
    if source_path.is_file():
        # Single file: return as list
        return [source_path]
    elif source_path.is_dir():
        # Directory: recursively search for all image and video files
        image_extensions = {'jpg', 'jpeg', 'png', 'bmp'}
        video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'm4v', '3gp', 'webm'}
        all_extensions = image_extensions | video_extensions
        
        media_files = []
        
        # Iterate through all items in directory
        for item in source_path.iterdir():
            if item.is_file():
                # Check if it's a media file
                ext = item.suffix.lower().lstrip('.')
                if ext in all_extensions:
                    media_files.append(item)
            elif item.is_dir():
                # Recursively search subdirectory (including symlinks)
                media_files.extend(get_media_files(item))
        
        return sorted(media_files)  # Sort for consistent ordering
    else:
        return []


def process_media_file(file_path, model_path, output_dir, file_idx, total_files):
    """
    Process a single media file (image or video).
    
    Args:
        file_path: Path to the media file
        model_path: Path to model file
        output_dir: Directory to save output
        file_idx: Current file index (for display)
        total_files: Total number of files (for display)
        
    Returns:
        tuple: (result_path, stats, is_video) if successful, (None, None, is_video) otherwise
    """
    is_video = is_video_file(file_path)
    file_type = "VIDEO" if is_video else "IMAGE"
    
    print(f"\n[{file_type} {file_idx}/{total_files}] Processing: {file_path.name}")
    print("-" * 50)
    
    if is_video:
        # NOTE: ONNX Runtime supports dynamic shapes, so rect=True works fine
        result_path, stats = run_video_inference(model_path, str(file_path), output_dir, debug=(DEBUG_MODE == 1), rect=RECT_OPT)
    else:
        # NOTE: ONNX Runtime supports dynamic shapes, so rect=True works fine
        result_path, stats = run_inference(model_path, str(file_path), output_dir, debug=(DEBUG_MODE == 1), rect=RECT_OPT)
    
    return result_path, stats, is_video


def main():
    """
    Main function to process single image, video, or directory of images/videos.
    Supports batch processing and provides detailed summary.
    """
    saved_files = []
    image_statistics = {}  # Store statistics for each image
    video_statistics = {}  # Store statistics for each video file
    
    source_path = Path(SOURCE_PATH)

    # Get list of media files
    media_files = get_media_files(source_path)
    
    if not media_files:
        print(f"No image or video files found in {source_path}")
        return
    
    # Count video and image files
    video_files = [f for f in media_files if is_video_file(f)]
    image_files = [f for f in media_files if not is_video_file(f)]
    
    # Print header
    if source_path.is_file():
        file_type = "video" if is_video_file(source_path) else "image"
        print(f"Processing single {file_type} file.")
    else:
        print("Processing directory of images and videos.")
    
    print(f"Letterbox mode: {'rect (preserve aspect ratio)' if RECT_OPT else 'square padding'}")
    print(f"Results will be saved in '{OUTPUT_DIR}' folder.")
    print(f"Found {len(image_files)} image(s) and {len(video_files)} video(s)")
    print("-" * 50)
    
    # Process all media files
    for idx, file_path in enumerate(media_files, 1):
        result_path, stats, is_video = process_media_file(
            file_path, MODEL_PATH, OUTPUT_DIR, idx, len(media_files)
        )
        
        if result_path:
            saved_files.append(result_path)
            if is_video:
                video_statistics[Path(result_path).name] = stats
            else:
                image_statistics[Path(result_path).name] = stats

    # Print summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {len(saved_files)}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Calculate total statistics
    total_inference_time = sum(stats['inference_time'] for stats in video_statistics.values() if stats)
    total_frames = sum(stats['frame_count'] for stats in video_statistics.values() if stats)
    
    print(f"\n{'='*80}")
    print("VIDEO PROCESSING STATISTICS")
    print(f"{'='*80}")
    
    for idx, file_path in enumerate(saved_files, 1):
        filename = Path(file_path).name
        file_type = "VIDEO" if is_video_file(file_path) else "IMAGE"
        
        print(f"\n[{idx}] {filename}")
        print(f"    Type: {file_type}")
        
        if file_type == "VIDEO" and filename in video_statistics:
            stats = video_statistics[filename]
            if stats:
                print(f"    Batch Size:               {stats['batch_size']}")
                print(f"    Frames Processed:         {stats['frame_count']}")
                print(f"    Overall Processing Time:  {stats['overall_time']:.2f}s")
                print(f"    Pure Inference Time:      {stats['inference_time']:.2f}s ({stats['inference_time']/stats['overall_time']*100:.1f}%)")
                print(f"    FPS (Overall):            {stats['fps']:.2f}")
                print(f"    FPS (Inference Only):     {stats['frame_count']/stats['inference_time']:.2f}" if stats['inference_time'] > 0 else "    FPS (Inference Only):  N/A")
    
    if video_statistics:
        # Calculate aggregate statistics
        total_overall_time = sum(stats['overall_time'] for stats in video_statistics.values() if stats)
        num_videos = len([f for f in saved_files if is_video_file(f)])
        inference_percentage = (total_inference_time / total_overall_time * 100) if total_overall_time > 0 else 0
        total_fps_overall = total_frames / total_overall_time if total_overall_time > 0 else 0
        total_fps_inference = total_frames / total_inference_time if total_inference_time > 0 else 0
        
        print(f"\n{'='*80}")
        print("AGGREGATE STATISTICS (All Videos)")
        print(f"{'='*80}")
        print(f"Total Files:                      {num_videos}")
        print(f"Batch Size:                       {BATCH_SIZE}")
        print(f"Total Frames Processed:           {total_frames}")
        print(f"Total Overall Processing Time:    {total_overall_time:.3f}s")
        print(f"Total Pure Inference Time:        {total_inference_time:.3f}s ({inference_percentage:.1f}%)")
        print(f"Total FPS (Overall):              {total_fps_overall:.2f}")
        print(f"Total FPS (Inference Only):       {total_fps_inference:.2f}")
        
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
        print(f"     ├─ Video I/O:         (frame reading, video writing)")
        print(f"     ├─ Visualization:     (drawing boxes, labels)")
        print(f"     └─ Misc:              (batching, progress display)")
        print(f"\nFPS Comparison:")
        print(f"  • FPS (Inference Only): {total_fps_inference:.2f} FPS")
        print(f"    → Pure model throughput (what the model can achieve)")
        print(f"    → Measures only: preprocessing + inference + postprocessing")
        print(f"")
        print(f"  • FPS (Overall):        {total_fps_overall:.2f} FPS")
        print(f"    → Real-world application performance")
        print(f"    → Includes all overheads: video I/O, visualization, etc.")
        print(f"")
        print(f"  • Performance Gap:      {total_fps_inference - total_fps_overall:.2f} FPS ({overhead_percentage:.1f}% overhead)")
        print(f"    → This gap represents non-inference operations")
        print(f"    → Typical range: 20-30% for video processing applications")
    
    print(f"\n{'='*80}")
    print("Processing completed successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()