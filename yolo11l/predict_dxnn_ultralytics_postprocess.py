"""
YOLOv11 DXNN Inference with Ultralytics Postprocessing

This implementation removes the dependency on Ultralytics YOLO class and provides
custom implementations for preprocessing and DXNN Runtime inference.
Only the postprocessing components (Results, ops, NMS) use Ultralytics utilities.

Implementation details:
- Preprocessing: Custom letterbox and image preprocessing
- Inference: Direct DXNN Runtime session execution
- Postprocessing: Ultralytics utilities (Results, ops.scale_boxes, non_max_suppression)

Dependencies: cv2, numpy, torch, dx-engine, ultralytics (postprocessing only)
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import torch

# Add ultralytics path
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

# Configuration
DEBUG_MODE = 1  # Set to 1 to enable debug output, 0 to disable

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_EXTENSION = 'dxnn'
MODEL_NAME = f'{CURRENT_DIR.name}'
MODEL_FILE = f'{CURRENT_DIR.name}.{MODEL_EXTENSION}'
MODEL_PATH = PROJECT_ROOT / MODEL_NAME / 'models' / MODEL_FILE
SOURCE_PATH = PROJECT_ROOT / 'assets' / 'images' / 'bus.jpg'      # for image file
# SOURCE_PATH = PROJECT_ROOT / 'assets' / 'images'                    # for image directory
OUTPUT_SUBDIR = CURRENT_DIR / 'runs' / 'predict' / MODEL_EXTENSION / "ultralytics_postprocess"
DEBUG_OUTPUT_DIR = OUTPUT_SUBDIR / 'debug'   # Directory to save debug outputs
OUTPUT_DIR = OUTPUT_SUBDIR  # Directory to save results

# Detection parameters (Ultralytics defaults)
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.25

INPUT_SIZE = 640

# Letterbox preprocessing mode
# False: Square padding (e.g., 640x640) - matches Ultralytics rect=False
# True: Rectangular, preserve aspect ratio (e.g., 480x640) - matches Ultralytics rect=True
# IMPORTANT: rect=False forces square padding (640x640) for DEEPX NPU fixed input shape
RECT_OPT = False

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

def letterbox(image, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32, rect=True):
    """
    Resize and pad image while meeting stride-multiple constraints.
    Custom implementation based on Ultralytics letterbox function.
    
    Args:
        image: Input image (numpy array)
        new_shape: Target size as (height, width) or int
        color: Padding color (RGB)
        auto: Enable stride-multiple auto-padding (minimum padding)
        scaleFill: Stretch image to fill new_shape (no padding, distorts aspect ratio)
        scaleup: Allow upscaling of image
        stride: Stride value for auto padding (default: 32)
        rect: Enable rectangular inference (preserve aspect ratio, no square padding)
              - True: Preserve aspect ratio (e.g., 480x640 for 640x480 input) - default
              - False: Force square padding (e.g., 640x640)
    
    Returns:
        image: Preprocessed image
        ratio: Scaling ratio as (ratio_w, ratio_h)
        (dw, dh): Padding as (pad_width, pad_height)
    """
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        # Minimum padding to meet stride-multiple constraints
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        # Stretch to fill (no padding, distorts aspect ratio)
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    elif rect:
        # Rectangular inference: preserve aspect ratio, no square padding
        # Round padding to stride multiple for efficient processing
        dw = np.mod(dw, stride)
        dh = np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)

def preprocess_image(image_path, imgsz=640, debug=False, rect=True):
    """
    Read image and preprocess it for model input.
    Custom preprocessing implementation without Ultralytics dependencies.
    
    Args:
        image_path: Path to input image
        imgsz: Target image size (default: 640)
        debug: Enable debug output
        rect: Enable rectangular inference (preserve aspect ratio)
              - True: Preserve aspect ratio (e.g., 480x640) - default
              - False: Force square padding (e.g., 640x640)
    
    Returns:
        processed_image: Preprocessed tensor with shape (1, 3, H, W)
        image: Original image (BGR)
        ratio: Scaling ratio
        (dw, dh): Padding
        preproc_shape: Actual preprocessing shape (H, W) after letterbox
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    original_height, original_width = image.shape[:2]
    
    # Apply letterbox preprocessing
    processed_image_bgr, ratio, (dw, dh) = letterbox(image, new_shape=(imgsz, imgsz), rect=rect)
    
    # Get actual preprocessing shape BEFORE normalization
    preproc_shape = processed_image_bgr.shape[:2]  # (H, W) after letterbox
    
    # Convert to RGB and normalize
    processed_image = cv2.cvtColor(processed_image_bgr, cv2.COLOR_BGR2RGB)
    processed_image = processed_image.transpose(2, 0, 1)  # HWC to CHW
    processed_image = np.ascontiguousarray(processed_image)
    processed_image = processed_image.astype(np.float32) / 255.0
    
    # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=0)
    
    if debug:
        print(f"Debug info:")
        print(f"  Original size: {original_width}x{original_height}")
        print(f"  Letterbox mode: {'rect (preserve aspect ratio)' if rect else 'square padding'}")
        print(f"  Preprocessing shape: {preproc_shape}")
        print(f"  Ratio: {ratio}")
        print(f"  Padding (dw, dh): {(dw, dh)}")
    print(f"\n[Preprocess] Input tensor shape: {processed_image.shape}")
    print(f"[Preprocess]  Preprocessing shape (H, W): {preproc_shape}")
    print(f"[Preprocess]  Input tensor range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    return processed_image, image, ratio, (dw, dh), preproc_shape

def postprocess_output(preds, orig_img, preproc_shape):
    """
    Postprocess Model output using Ultralytics utilities.
    Uses Ultralytics non_max_suppression, ops.scale_boxes, and Results class.
    
    Args:
        preds: Raw Model output tensor [1, 84, 8400] for YOLO11
        orig_img: Original image (BGR) - used for shape extraction and Results object creation

    Returns:
        Results: Results object with box detections
    """
    # Convert to torch tensor if needed
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).float()
    
    print(f"\n[Postprocess] Input shape: {preds.shape}")
    print(f"[Postprocess] Input range: [{preds.min():.3f}, {preds.max():.3f}]")
    
    # Apply Non-Maximum Suppression using Ultralytics
    # Ultralytics NMS expects [batch, num_classes + 4, num_detections]
    # Current format: [1, 84, 8400] where 84 = 4(bbox) + 80(classes)
    # This is already in the correct format - NO transpose needed!
    results = non_max_suppression(
        preds,
        conf_thres=CONFIDENCE_THRESHOLD,
        iou_thres=IOU_THRESHOLD,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=80,  # 80 classes for COCO dataset
    )
    
    print(f"[Postprocess] NMS output: {len(results)} image(s)")
    if len(results) > 0 and len(results[0]) > 0:
        print(f"[Postprocess] First result shape: {results[0].shape}")
        print(f"[Postprocess] Total detections: {len(results[0])}")
    
    # Process first image result
    if len(results) == 0 or len(results[0]) == 0:
        # No detections
        return Results(
            orig_img=orig_img,
            path=None,
            names={i: name for i, name in enumerate(CLASSES)},
            boxes=None
        )
    
    pred = results[0]  # [num_detections, 6] = [x1, y1, x2, y2, conf, cls]
    
    print(f"[Postprocess] Pred shape before scale: {pred.shape}")
    print(f"[Postprocess] Pred sample (first detection): {pred[0] if len(pred) > 0 else 'none'}")
    
    # Scale boxes from letterbox size to original image size
    # Input size is 640x640
    print(f"[Postprocess] Using preprocessing shape: {preproc_shape}")
    pred[:, :4] = ops.scale_boxes(preproc_shape, pred[:, :4], orig_img.shape)
    
    print(f"[Postprocess] Box data shape after scale: {pred.shape}")
    print(f"[Postprocess] Confidence range: {pred[:, 4].min():.3f} ~ {pred[:, 4].max():.3f}")
    
    # Create Results object
    # Results expects boxes in format: [n, 6] where each row is [x1, y1, x2, y2, conf, cls]
    result = Results(
        orig_img=orig_img,
        path=None,
        names={i: name for i, name in enumerate(CLASSES)},
        boxes=pred  # [n, 6] tensor: [x1, y1, x2, y2, conf, cls]
    )
    
    return result

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

def run_inference_using_dxnn(model_path, input_tensor):
    """
    Run inference with DXNN model using dx_engine.
    """
    if not isinstance(model_path, str):
        model_path = str(model_path)

    # Initialize InferenceEngine
    from dx_engine import InferenceEngine
    engine = InferenceEngine(model_path)
    
    # Convert input tensor to uint8 format as expected by DEEPX
    # input_tensor is in range [0, 1], convert to [0, 255]
    im_np = (input_tensor * 255).astype("uint8")
    
    # Convert from NCHW (Batch, Channel, Height, Width) to HWC (Height, Width, Channel)
    if len(im_np.shape) == 4:  # NCHW format
        im_np = np.squeeze(im_np, axis=0)  # Remove batch dimension (N=1)
        im_np = np.transpose(im_np, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
    
    # Prepare input data as list
    input_data = [im_np]
    
    # Run inference using DEEPX InferenceEngine
    outputs = engine.run(input_data)
    
    return outputs

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
        str: Path to output image if successful, None otherwise
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

        # Process image
        print(f"\nProcessing: {Path(image_path).name}")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Preprocess
        input_tensor, orig_img, ratio, pad, preproc_shape = preprocess_image(image_path, INPUT_SIZE, debug=debug, rect=rect)

        # Debug: Visualize preprocessed input tensor
        if debug:
            preprocess_image_dir = Path(DEBUG_OUTPUT_DIR) / 'input'
            preprocess_image_dir.mkdir(parents=True, exist_ok=True)
            preprocess_image_path = Path(preprocess_image_dir) / f'preprocessed_input_{Path(image_path).stem}_{timestamp}.jpg'
            debug_visualize_tensor(input_tensor, title="Preprocessed Input Tensor", save_path=preprocess_image_path, show=False)
        
        # 2. Inference
        outputs = run_inference_using_dxnn(model_path, input_tensor)
        preds = outputs[0]  # Get first (and only) result

        for idx, output in enumerate(outputs):
            print(f"Raw output[{idx}] shape: {output.shape}")
            print(f"Raw output[{idx}] range: [{output.min():.3f}, {output.max():.3f}]")
            
            # Debug: Save raw output for debugging
            if debug:
                raw_output_dir = Path(DEBUG_OUTPUT_DIR) / 'raw_output'
                raw_output_dir.mkdir(parents=True, exist_ok=True)
                raw_output_path = Path(raw_output_dir) / f'raw_output{idx}_{Path(image_path).stem}_{timestamp}.npy'
                np.save(str(raw_output_path), output)
                print(f"Inference Raw output saved to: {raw_output_path}")

        # 3. Post-processing
        result = postprocess_output(preds, orig_img, preproc_shape)

        # 4. Visualization and analysis
        filename = Path(image_path).stem
        output_filename = Path(image_path).stem + f'_detected_{timestamp}.jpg'
        output_path = str(Path(output_dir) / output_filename)

        # Draw detections
        draw_detections(image_path, result, output_path, save=save, show=show)

        # Print analysis result
        if debug:
            analyze_results(result, filename)

        return output_path

    except Exception as e:
        print(f"[{Path(image_path).name}] Error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def debug_visualize_tensor(tensor, title="Input Tensor", save_path=None, show=False):
    """
    Debug utility to visualize tensors as images.
    Converts NCHW/CHW format tensors to displayable BGR images.
    """
    try:
        # Convert tensor to numpy if needed
        if isinstance(tensor, torch.Tensor):
            im_vis = tensor.cpu().numpy()
        else:
            im_vis = tensor
        
        # Handle different tensor formats
        if len(im_vis.shape) == 4:  # NCHW format
            im_vis = im_vis.squeeze(0).transpose(1, 2, 0)  # (H, W, C)
        elif len(im_vis.shape) == 3:  # CHW format
            im_vis = im_vis.transpose(1, 2, 0)  # (H, W, C)
        
        # Convert to displayable format
        if im_vis.dtype == np.float32 or im_vis.dtype == np.float64:
            # Convert from [0, 1] range to [0, 255]
            im_vis = (im_vis * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if im_vis.shape[-1] == 3:
            im_vis = im_vis[:, :, ::-1]
        
        # Show image
        if show:
            cv2.imshow(title, im_vis)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, im_vis)
            print(f"Tensor visualization saved to: {save_path}")
        
        print(f"[Debug] {title} - Shape: {tensor.shape if isinstance(tensor, torch.Tensor) else im_vis.shape}")
        return im_vis
        
    except Exception as e:
        print(f"[Debug] Failed to visualize tensor: {e}")
        print(f"[Debug] Tensor info - Type: {type(tensor)}, Shape: {tensor.shape if hasattr(tensor, 'shape') else 'unknown'}")

def main():
    """
    Main function to process single image or directory of images.
    Supports batch processing and provides detailed summary.
    """
    saved_files = []
    
    source_path = Path(SOURCE_PATH)

    # Check if source is file or directory
    if source_path.is_file():
        print("Processing single image file.")
        print(f"Letterbox mode: {'rect (preserve aspect ratio)' if RECT_OPT else 'square padding'}")
        print(f"Results will be saved in '{OUTPUT_DIR}' folder.")
        print("-" * 50)
        # IMPORTANT: rect=False forces square padding (640x640) for DEEPX NPU fixed input shape
        result_path = run_inference(MODEL_PATH, str(source_path), OUTPUT_DIR, debug=(DEBUG_MODE == 1), rect=RECT_OPT)
        if result_path:
            saved_files.append(result_path)

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
            # IMPORTANT: rect=False forces square padding (640x640) for DEEPX NPU fixed input shape
            result_path = run_inference(MODEL_PATH, str(image_path), OUTPUT_DIR, debug=(DEBUG_MODE == 1), rect=RECT_OPT)
            if result_path:
                saved_files.append(result_path)
    else:
        print(f"Error: {source_path} is neither a file nor a directory")
        return

    # Print summary
    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total images processed: {len(saved_files)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nSaved files:")
    for idx, file_path in enumerate(saved_files, 1):
        print(f"  {idx}. {Path(file_path)}")
    print(f"{'='*70}")
    print("Processing completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()