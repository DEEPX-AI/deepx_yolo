# DEEPX-YOLO Project

This project demonstrates various YOLO Model inference implementations, from standalone ONNX Runtime to DEEPX-accelerated execution. It provides multiple example scripts for different use cases: object detection, pose estimation, instance segmentation, and oriented bounding box detection.

## 🎯 Project Overview

This repository includes:
- **Standalone implementations**: Zero external dependencies (ONNX/DXNN inference without Ultralytics)
- **Ultralytics DEEPX implementations**: Using custom Ultralytics DEEPX library for enhanced debugging and DXNN support
- **Multiple task support**: Detection, Pose Estimation, Segmentation, OBB
- **Model conversion utilities**: PyTorch to ONNX export scripts

## 🗂️ Directory Structure

```plaintext
yolov11l_poc/
├── README.md
├── requirements.txt
├── assets/                     # Sample images
│   ├── boats.jpg
│   ├── bus.jpg
│   └── zidane.jpg
├── test_images/               # Test images folder
│   └── 1.jpg ~ 7.jpg
├── lib/
│   └── ultralytics/           # Custom Ultralytics DEEPX library (submodule)
├── yolo11l/                   # Object Detection examples
│   ├── export_onnx.py                            # PyTorch to ONNX conversion
│   ├── ultralytics_deepx_lib_setup.py            # Custom library setup script
│   ├── predict_onnx_standalone.py                # Standalone ONNX inference (zero dependencies)
│   ├── predict_onnx_ultralytics_postprocess.py   # ONNX inference + Ultralytics postprocessing (hybrid)
│   ├── predict_onnx_ultralytics_deepx.py         # ONNX inference with custom Ultralytics DEEPX library
│   ├── predict_dxnn_standalone.py                # Standalone DXNN inference (zero dependencies)
│   ├── predict_dxnn_ultralytics_postprocess.py   # DXNN inference + Ultralytics postprocessing (hybrid)
│   ├── predict_dxnn_ultralytics_deepx.py         # DXNN inference with custom Ultralytics DEEPX library
│   ├── models/
│   │   ├── metadata.yaml
│   │   ├── yolo11l.pt                        # PyTorch model
│   │   ├── yolo11l.onnx                      # ONNX model
│   │   └── yolo11l.dxnn                      # DEEPX model
│   └── runs/predict/                         # Output results
├── yolo11l-pose/              # Pose estimation example
│   ├── ultralytics_deepx_lib_setup.py
│   ├── export_onnx.py
│   ├── predict_onnx_standalone.py
│   ├── predict_onnx_ultralytics_postprocess.py
│   ├── predict_onnx_ultralytics_deepx.py
│   ├── predict_dxnn_standalone.py
│   ├── predict_dxnn_ultralytics_postprocess.py
│   ├── predict_dxnn_ultralytics_deepx.py
│   └── models/
├── yolo11l-seg/               # Instance segmentation example
│   ├── ultralytics_deepx_lib_setup.py
│   ├── export_onnx.py
│   ├── predict_onnx_standalone.py
│   ├── predict_onnx_ultralytics_postprocess.py
│   ├── predict_onnx_ultralytics_deepx.py
│   ├── predict_dxnn_standalone.py
│   ├── predict_dxnn_ultralytics_postprocess.py
│   ├── predict_dxnn_ultralytics_deepx.py
│   └── models/
└── yolo11l-obb/               # Oriented bounding box example
│   ├── ultralytics_deepx_lib_setup.py
│   ├── export_onnx.py
│   ├── predict_onnx_standalone.py
│   ├── predict_onnx_ultralytics_postprocess.py
│   ├── predict_onnx_ultralytics_deepx.py
│   ├── predict_dxnn_standalone.py
│   ├── predict_dxnn_ultralytics_postprocess.py
│   ├── predict_dxnn_ultralytics_deepx.py
    └── models/
```

## 🛠️ Prerequisites

### 1. Python Environment Requirements

- Python 3.12 or higher (tested with Python 3.12.3)

### 2. Required Package Installation

```bash
# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux
# venv\Scripts\activate   # On Windows

# Install required packages
pip install -r requirements.txt
```

### 3. Main Dependencies

**Core Libraries:**
- **torch**: PyTorch for tensor operations
- **ultralytics**: YOLOv11 model loading and conversion (for Ultralytics DEEPX examples)
- **opencv-python**: Image processing and visualization
- **numpy**: Numerical computation
- **onnxruntime**: ONNX model inference

**DEEPX Runtime Python Library (for DXNN inference):**
- **dx-engine**: DEEPX runtime for DXNN model inference

### 4. Custom Ultralytics DEEPX Library Setup

The custom Ultralytics DEEPX library is included as a Git submodule in `lib/ultralytics/`. It provides:
- Debug visualization of input tensors
- Debug saving of raw output tensors
- DXNN model inference support

To initialize the submodule:
```bash
git submodule update --init --recursive
```

The `ultralytics_deepx_lib_setup.py` script automatically configures the Python path to use this custom library.

## 📥 Model Download

### YOLOv11 Model Download

YOLOv11 models can be downloaded from [Ultralytics official documentation](https://docs.ultralytics.com/models/yolo11/#performance-metrics).

```bash
# Navigate to yolo11l folder
cd yolo11l/models

# Download YOLOv11l model (approximately 50MB)
# Method 1: Using wget
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt

# Method 2: Direct download
# Download from browser and save to yolo11l/models/ folder
```

**Available Models:**
- `yolo11l.pt`: Object Detection (PyTorch format)
- `yolo11l-pose.pt`: Pose Estimation
- `yolo11l-seg.pt`: Instance Segmentation
- `yolo11l-obb.pt`: Oriented Bounding Box Detection

## 🚀 Usage

### 1. Model Conversion (PyTorch → ONNX)

```bash
cd yolo11l
python export_onnx.py
```

**export_onnx.py features:**
- Converts `models/yolo11l.pt` → `models/yolo11l.onnx`
- Exports metadata.yaml with model configuration
- Uses ONNX opset 21 for maximum compatibility
- Supports SSL certificate bypass for corporate environments

### 2. Object Detection Inference

The project provides **six different inference implementations**:

#### 2.1. Standalone ONNX Inference (Recommended for learning)

```bash
cd yolo11l
python predict_onnx_standalone.py
```

**Features:**
- ✅ **Zero Ultralytics dependencies** - All functions ported into single file
- ✅ **Fully self-contained** - Complete Boxes and Results classes included
- ✅ **Educational** - Easy to understand preprocessing, inference, and postprocessing
- ✅ **Reusable** - Can be adapted for pose, segmentation, and OBB tasks

**Implementation highlights:**
- Custom preprocessing: letterbox, normalization
- Direct ONNX Runtime execution
- Ported NMS and coordinate scaling functions
- Complete Results object with all Boxes attributes

#### 2.2. ONNX Inference with Ultralytics Postprocessing (Hybrid)

```bash
cd yolo11l
python predict_onnx_ultralytics_postprocess.py
```

**Features:**
- ✅ **Hybrid Approach** - Custom preprocessing/inference + Ultralytics postprocessing
- ✅ **Minimal Dependencies** - Uses Ultralytics utilities only for postprocessing
- ✅ **Flexibility** - Custom preprocessing/inference with validated postprocessing
- ✅ **Learning-Friendly** - Clear separation of each step

**Implementation highlights:**
- Custom preprocessing: letterbox, normalization
- Direct ONNX Runtime execution
- Ultralytics postprocessing: non_max_suppression, ops.scale_boxes, Results class
- Uses Ultralytics utilities without YOLO class

#### 2.3. ONNX Inference with Ultralytics DEEPX Library

```bash
cd yolo11l
python predict_onnx_ultralytics_deepx.py
```

**Features:**
- Uses custom Ultralytics DEEPX library
- Complete YOLO class for end-to-end inference
- Debug features: input tensor visualization, raw output saving
- All preprocessing/inference/postprocessing handled by Ultralytics

#### 2.4. Standalone DXNN Inference

```bash
cd yolo11l
python predict_dxnn_standalone.py
```

**Features:**
- DEEPX runtime for accelerated inference
- Zero Ultralytics dependencies
- Similar structure to standalone ONNX version

#### 2.5. DXNN Inference with Ultralytics Postprocessing (Hybrid)

```bash
cd yolo11l
python predict_dxnn_ultralytics_postprocess.py
```

**Features:**
- ✅ **Hybrid Approach** - Custom preprocessing/DXNN inference + Ultralytics postprocessing
- ✅ **DEEPX Acceleration** - Fast inference via DXNN runtime
- ✅ **Validated Postprocessing** - Uses Ultralytics NMS and coordinate scaling
- ✅ **Production-Ready** - Balance of performance and accuracy

**Implementation highlights:**
- Custom preprocessing: letterbox, normalization
- DXNN Runtime execution (dx_engine)
- Ultralytics postprocessing: non_max_suppression, ops.scale_boxes, Results class
- Leverages DEEPX acceleration without full YOLO class

#### 2.6. DXNN Inference with Ultralytics DEEPX Library

```bash
cd yolo11l
python predict_dxnn_ultralytics_deepx.py
```

**Features:**
- Uses DEEPX runtime via custom Ultralytics library
- Complete YOLO class support
- Enhanced debugging capabilities

### 3. Execution Process

All inference scripts follow this pattern:

1. **Input**: Search for images in `../assets/` or specified directory
2. **Preprocessing**: Letterbox resize, normalization, channel conversion
3. **Inference**: ONNX Runtime or DEEPX execution
4. **Postprocessing**: NMS, coordinate scaling, Results object creation
5. **Visualization**: Draw bounding boxes and save results
6. **Output**: Save to `runs/predict/{backend}/{script_name}/` directory

### 4. Execution Result Example

```plaintext
Processing single image file.
Letterbox mode: square padding
Results will be saved in '/data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/dxnn/standalone' folder.
--------------------------------------------------

Processing: bus.jpg
Debug info:
  Original size: 810x1080
  Letterbox mode: square padding
  Preprocessing shape: (640, 640)
  Ratio: (0.5925925925925926, 0.5925925925925926)
  Padding (dw, dh): (80.0, 0.0)

[Preprocess] Input tensor shape: (1, 3, 640, 640)
[Preprocess]  Preprocessing shape (H, W): (640, 640)
[Preprocess]  Input tensor range: [0.000, 1.000]
Tensor visualization saved to: /data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/dxnn/standalone/debug/input/preprocessed_input_bus_20251106_101311_990.jpg
[Debug] Preprocessed Input Tensor - Shape: (640, 640, 3)
/data/home/dhyang/git/deepx_yolo/venv/lib/python3.12/site-packages/dx_engine/utils.py:23: UserWarning: ndarray(shape=(640, 640, 3), dtype=uint8) is not contiguous; converting.
  warnings.warn(
Raw output[0] shape: (1, 84, 8400)
Raw output[0] range: [0.000, 656.422]
Inference Raw output saved to: /data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/dxnn/standalone/debug/raw_output/raw_output0_bus_20251106_101311_990.npy

[Postprocess] Input shape: torch.Size([1, 84, 8400])
[Postprocess] Input range: [0.000, 656.422]
[Postprocess] NMS output: 1 image(s)
[Postprocess] First result shape: torch.Size([5, 6])
[Postprocess] Total detections: 5
[Postprocess] Pred shape before scale: torch.Size([5, 6])
[Postprocess] Pred sample (first detection): tensor([ 81.5273, 133.9609, 554.6602, 442.5391,   0.9299,   5.0000])
[Postprocess] Using preprocessing shape: (640, 640)
[Postprocess] Box data shape after scale: torch.Size([5, 6])
[Postprocess] Confidence range: 0.849 ~ 0.930
Detection result saved to '/data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/dxnn/standalone/bus_detected_20251106_101311_990.jpg' file.
==================================================
Total object detections: 5
Boxes tensor shape: torch.Size([5, 6])
Confidence range: 0.849 ~ 0.930
Confidences >= 0.25: 5
Class distribution: {np.int64(0): np.int64(4), np.int64(5): np.int64(1)}
Conf 0.2~0.3: 0
Conf 0.3~0.4: 0
Conf 0.4~0.5: 0
Conf 0.5~0.6: 0
Conf 0.6~0.7: 0
Conf 0.7~0.8: 0
Conf 0.8~0.9: 2
Conf >= 0.9: 3
[bus] Total 5 objects detected.
  1. bus: 0.93 - Position: (3, 226) ~ (801, 747)
  2. person: 0.90 - Position: (224, 411) ~ (345, 865)
  3. person: 0.90 - Position: (51, 406) ~ (249, 902)
  4. person: 0.89 - Position: (666, 399) ~ (810, 877)
  5. person: 0.85 - Position: (1, 555) ~ (81, 872)

======================================================================
PROCESSING SUMMARY
======================================================================
Total images processed: 1
Output directory: /data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/dxnn/standalone

Saved files:
  1. /data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/dxnn/standalone/bus_detected_20251106_101311_990.jpg
======================================================================
Processing completed successfully!
======================================================================
```

### 5. Video (Batch) Processing Execution Result Example (predict_onnx_ultralytics_deepx_video.py)

```plaintext
Added custom ultralytics path: /data/home/dhyang/git/deepx_yolo/yolo11l/../lib/ultralytics
Processing directory of images and videos.
Letterbox mode: rect (preserve aspect ratio)
Results will be saved in '/data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/onnx/ultralytics_deepx' folder.
Found 0 image(s) and 3 video(s)
--------------------------------------------------

[VIDEO 1/3] Processing: dance-group2.mov
--------------------------------------------------

Processing video: dance-group2.mov
Video properties: 1920x1080, 25 FPS, 764 frames
Output video will be saved to: /data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/onnx/ultralytics_deepx/dance-group2_detected_20251106_094834_268.mp4
Starting video processing with batch size: 12
--------------------------------------------------
Loading /data/home/dhyang/git/deepx_yolo/yolo11l/models/yolo11l.onnx for ONNX Runtime inference...
Using ONNX Runtime 1.23.2 CPUExecutionProvider
Processing frames 1-12/764 (1.6%) | Batch: 1.06s, 11.33 FPS | Cumulative: 11.33 FPS
...
Processing frames 745-756/764 (99.0%) | Batch: 0.77s, 15.51 FPS | Cumulative: 14.78 FPS

Video processing completed!
Processed 764/764 frames
Total time: 60.28s, Pure inference time: 51.64s
FPS (Overall): 12.68, FPS (Inference Only): 14.79
Output video saved to: /data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/onnx/ultralytics_deepx/dance-group2_detected_20251106_094834_268.mp4

[VIDEO 2/3] Processing: dogs.mp4
--------------------------------------------------

Processing video: dogs.mp4
Video properties: 1920x1080, 30 FPS, 300 frames
Output video will be saved to: /data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/onnx/ultralytics_deepx/dogs_detected_20251106_094934_544.mp4
Starting video processing with batch size: 12
--------------------------------------------------
Loading /data/home/dhyang/git/deepx_yolo/yolo11l/models/yolo11l.onnx for ONNX Runtime inference...
Using ONNX Runtime 1.23.2 CPUExecutionProvider
Processing frames 1-12/300 (4.0%) | Batch: 1.00s, 12.01 FPS | Cumulative: 12.01 FPS
...
Processing frames 289-300/300 (100.0%) | Batch: 0.81s, 14.74 FPS | Cumulative: 14.49 FPS

Video processing completed!
Processed 300/300 frames
Total time: 23.98s, Pure inference time: 20.70s
FPS (Overall): 12.51, FPS (Inference Only): 14.49
Output video saved to: /data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/onnx/ultralytics_deepx/dogs_detected_20251106_094934_544.mp4

[VIDEO 3/3] Processing: dron-citry-road2.mov
--------------------------------------------------

Processing video: dron-citry-road2.mov
Video properties: 1920x1080, 29 FPS, 210 frames
Output video will be saved to: /data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/onnx/ultralytics_deepx/dron-citry-road2_detected_20251106_094958_523.mp4
Starting video processing with batch size: 12
--------------------------------------------------
Loading /data/home/dhyang/git/deepx_yolo/yolo11l/models/yolo11l.onnx for ONNX Runtime inference...
Using ONNX Runtime 1.23.2 CPUExecutionProvider
Processing frames 1-12/210 (5.7%) | Batch: 0.98s, 12.19 FPS | Cumulative: 12.19 FPS
...
Processing frames 193-204/210 (97.1%) | Batch: 0.87s, 13.76 FPS | Cumulative: 14.52 FPS

Video processing completed!
Processed 210/210 frames
Total time: 17.39s, Pure inference time: 14.43s
FPS (Overall): 12.08, FPS (Inference Only): 14.56
Output video saved to: /data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/onnx/ultralytics_deepx/dron-citry-road2_detected_20251106_094958_523.mp4

================================================================================
PROCESSING SUMMARY
================================================================================
Total files processed: 3
Output directory: /data/home/dhyang/git/deepx_yolo/yolo11l/runs/predict/onnx/ultralytics_deepx

================================================================================
VIDEO PROCESSING STATISTICS
================================================================================

[1] dance-group2_detected_20251106_094834_268.mp4
    Type: VIDEO
    Batch Size:               12
    Frames Processed:         764
    Overall Processing Time:  60.28s
    Pure Inference Time:      51.64s (85.7%)
    FPS (Overall):            25.00
    FPS (Inference Only):     14.79

[2] dogs_detected_20251106_094934_544.mp4
    Type: VIDEO
    Batch Size:               12
    Frames Processed:         300
    Overall Processing Time:  23.98s
    Pure Inference Time:      20.70s (86.3%)
    FPS (Overall):            30.00
    FPS (Inference Only):     14.49

[3] dron-citry-road2_detected_20251106_094958_523.mp4
    Type: VIDEO
    Batch Size:               12
    Frames Processed:         210
    Overall Processing Time:  17.39s
    Pure Inference Time:      14.43s (83.0%)
    FPS (Overall):            29.00
    FPS (Inference Only):     14.56

================================================================================
AGGREGATE STATISTICS (All Videos)
================================================================================
Total Files:                      3
Batch Size:                       12
Total Frames Processed:           1274
Total Overall Processing Time:    101.640s
Total Pure Inference Time:        86.770s (85.4%)
Total FPS (Overall):              12.53
Total FPS (Inference Only):       14.68

================================================================================
PERFORMANCE BREAKDOWN EXPLANATION
================================================================================

Total Processing Time Breakdown:
  ├─ Pure Inference Time:  86.77s (85.4%)
  │  ├─ Preprocessing:     (letterbox, normalization)
  │  ├─ Inference:         (NPU/GPU execution)
  │  └─ Postprocessing:    (NMS, coordinate scaling)
  │
  └─ Overhead Time:        14.87s (14.6%)
     ├─ Video I/O:         (frame reading, video writing)
     ├─ Visualization:     (drawing boxes, labels)
     └─ Misc:              (batching, progress display)

FPS Comparison:
  • FPS (Inference Only): 14.68 FPS
    → Pure model throughput (what the model can achieve)
    → Measures only: preprocessing + inference + postprocessing

  • FPS (Overall):        12.53 FPS
    → Real-world application performance
    → Includes all overheads: video I/O, visualization, etc.

  • Performance Gap:      2.15 FPS (14.6% overhead)
    → This gap represents non-inference operations
    → Typical range: 20-30% for video processing applications

================================================================================
Processing completed successfully!
================================================================================
```



## ⚙️ Configuration Options

### Common Configuration (All Scripts)

```python
# Model paths
MODEL_PATH = 'models/yolo11l.onnx'  # or 'models/yolo11l.dxnn' for DXNN
SOURCE_PATH = '../assets'            # Input image path (file or directory)
OUTPUT_DIR = 'runs/predict/...'     # Result storage directory

# Detection parameters (Ultralytics defaults)
CONFIDENCE_THRESHOLD = 0.25   # Confidence threshold (0.0 ~ 1.0)
IOU_THRESHOLD = 0.45         # IoU threshold for NMS
INPUT_SIZE = 640             # Model input size
```



## 🔧 Troubleshooting

### SSL Certificate Error


### ONNX Runtime Error

```bash
# Reinstall CPU version
pip uninstall onnxruntime
pip install onnxruntime

# For GPU version (requires NVIDIA GPU)
pip install onnxruntime-gpu
```

### Custom Ultralytics Library Not Found

```bash
# Initialize Git submodule
git submodule update --init --recursive

# Verify lib/ultralytics/ exists
ls lib/ultralytics/

# The ultralytics_deepx_lib_setup.py script should handle path configuration
```

### DEEPX Runtime Error

```bash
# Install dx-engine for DXNN inference
pip install dx-engine

# Verify installation
python -c "from dx_engine import InferenceEngine; print('DEEPX OK')"
```

## 📈 Performance Information

### Model Specifications

| Model | Size | mAP50-95 | Speed (CPU) | Parameters | FLOPs |
|-------|------|----------|-------------|------------|-------|
| YOLOv11n | ~6MB | 39.5% | ~50ms | 2.6M | 6.5B |
| YOLOv11s | ~19MB | 47.0% | ~100ms | 9.4M | 21.5B |
| YOLOv11m | ~40MB | 51.5% | ~200ms | 20.1M | 68.0B |
| YOLOv11l | ~50MB | 53.4% | ~300ms | 25.3M | 86.9B |
| YOLOv11x | ~110MB | 54.7% | ~500ms | 56.9M | 194.9B |

### Inference Performance

**ONNX Runtime (CPU):**
- Image preprocessing: ~10-20ms
- Model inference: ~200-500ms (depends on model size)
- Postprocessing (NMS): ~10-30ms
- Total: ~250-550ms per image

**DEEPX Runtime (Accelerated):**
- Significant speedup on supported hardware
- Optimized memory usage
- Lower latency for batch processing

### Supported Resolutions

- Default input: 640x640 (automatic letterbox resize)
- Maximum tested: 1920x1080
- Minimum recommended: 320x320

## 🎓 Learning Resources

### Understanding the Code

1. **Start with standalone scripts**: 
   - `predict_onnx_standalone.py` is the best starting point
   - All preprocessing, inference, and postprocessing in one file
   - Well-commented with debug outputs

2. **Key concepts to understand**:
   - **Letterbox preprocessing**: Maintain aspect ratio while resizing
   - **NMS (Non-Maximum Suppression)**: Remove duplicate detections
   - **Coordinate scaling**: Convert from model space to image space
   - **Results object**: Container for all detection information

3. **Progression path**:
   ```
   predict_onnx_standalone.py                  → Understand full pipeline
   predict_onnx_ultralytics_postprocess.py     → Learn hybrid approach
   predict_onnx_ultralytics_deepx.py           → See Ultralytics DEEPX integration library (image)
   predict_onnx_ultralytics_deepx_video.py     → Ultralytics DEEPX video processing
   predict_dxnn_standalone.py                  → Learn DEEPX runtime
   predict_dxnn_ultralytics_postprocess.py     → DEEPX + validated postprocessing
   predict_dxnn_ultralytics_deepx.py           → See Ultralytics DEEPX integration library (image)
   predict_dxnn_ultralytics_deepx_video.py     → Ultralytics DEEPX video processing (DXNN accelerated)
   ```

### Code Reuse

standalone scripts include complete implementations that can be reused:

```python
# From predict_onnx_standalone.py

# Reusable Boxes class with 11 attributes
class Boxes:
    - xyxy, xywh, xyxyn, xywhn  # Various coordinate formats
    - conf, cls, id              # Detection metadata
    - shape, is_track            # Properties
    - cpu(), numpy(), cuda()     # Device management

# Reusable utility functions
def letterbox()              # Aspect-ratio preserving resize
def preprocess_image()       # Complete preprocessing pipeline
def non_max_suppression()    # NMS implementation
def scale_boxes()            # Coordinate transformation
```

These can be adapted for:
- Pose estimation (add keypoint processing)
- Instance segmentation (add mask processing)
- Oriented bounding boxes (add angle processing)

## File Descriptions

### Core Scripts (yolo11l/)

| File | Purpose | Dependencies | Use Case |
|------|---------|--------------|----------|
| `export_onnx.py` | Convert PyTorch to ONNX | ultralytics, torch | Model preparation |
| `ultralytics_deepx_lib_setup.py` | Configure custom library path | - | Library initialization |
| `predict_onnx_standalone.py` | ONNX inference (zero deps) | cv2, numpy, torch, onnxruntime | Learning, customization |
| `predict_onnx_ultralytics_postprocess.py` | ONNX inference + Ultralytics postprocess | cv2, numpy, torch, onnxruntime, ultralytics (postprocess only) | Hybrid approach, validated postprocessing |
| `predict_onnx_ultralytics_deepx.py` | ONNX inference (image, full lib) | ultralytics (custom), cv2, numpy, torch | Quick prototyping, debugging |
| `predict_onnx_ultralytics_deepx_video.py` | ONNX inference (video, full lib) | ultralytics (custom), cv2, numpy, torch | Video processing, frame-by-frame inference |
| `predict_dxnn_standalone.py` | DXNN inference (zero deps) | cv2, numpy, torch, dx-engine | Production, minimal deps |
| `predict_dxnn_ultralytics_postprocess.py` | DXNN inference + Ultralytics postprocess | cv2, numpy, torch, dx-engine, ultralytics (postprocess only) | DEEPX acceleration + validated postprocessing |
| `predict_dxnn_ultralytics_deepx.py` | DXNN inference (image, full lib) | ultralytics (custom), dx-engine | Development with acceleration |
| `predict_dxnn_ultralytics_deepx_video.py` | DXNN inference (video, full lib) | ultralytics (custom), dx-engine | DXNN accelerated video processing |

### Output Structure

```plaintext
yolo11l/runs/predict/
├── onnx/
│   ├── standalone/                    # predict_onnx_standalone.py outputs
│   │   ├── [input_image_name]_detected_[timestamp].jpg
│   │   └── debug/
│   │       ├── input/                 # Preprocessed input visualizations
│   │       └── raw_output/            # Raw model outputs (.npy)
│   ├── ultralytics_postprocess/       # predict_onnx_ultralytics_postprocess.py outputs
│   │   ├── [input_image_name]_detected_[timestamp].jpg
│   │   └── debug/
│   │       ├── input/                 # Preprocessed input visualizations
│   │       └── raw_output/            # Raw model outputs (.npy)
│   └── ultralytics_deepx/             # predict_onnx_ultralytics_deepx.py & _video.py outputs
│       ├── [input_image_name]_detected_[timestamp].jpg      # Image outputs
│       ├── [input_video_name]_detected_[timestamp].mp4      # Video outputs
│       └── debug/
│           ├── input/                 # Preprocessed input visualizations
│           ├── raw_output/            # Raw model outputs (.npy)
│           └── origin_output/         # Ultralytics native outputs
└── dxnn/
    ├── standalone/                    # predict_dxnn_standalone.py outputs
    │   ├── [input_image_name]_detected_[timestamp].jpg
    │   └── debug/
    │       ├── input/                 # Preprocessed input visualizations
    │       └── raw_output/            # Raw model outputs (.npy)
    ├── ultralytics_postprocess/       # predict_dxnn_ultralytics_postprocess.py outputs
    │   ├── [input_image_name]_detected_[timestamp].jpg
    │   └── debug/
    │       ├── input/                 # Preprocessed input visualizations
    │       └── raw_output/            # Raw model outputs (.npy)
    └── ultralytics_deepx/             # predict_dxnn_ultralytics_deepx.py & _video.py outputs
        ├── [input_image_name]_detected_[timestamp].jpg      # Image outputs
        ├── [input_video_name]_detected_[timestamp].mp4      # Video outputs
        └── debug/
            ├── input/                 # Preprocessed input visualizations
            ├── raw_output/            # Raw model outputs (.npy)
            └── origin_output/         # Ultralytics native outputs
    
```


## 🔍 Debugging and Comparison

### Comparing Outputs

The project includes utilities to compare outputs between different implementations:

```bash
# Compare raw model outputs (standalone vs Ultralytics DEEPX)
python util/compare_raw_outputs.py \
    runs/predict/onnx/standalone/debug/raw_output/raw_output_[timestamp].npy \
    runs/predict/onnx/ultralytics_deepx/debug/raw_output/raw_output_[timestamp].npy

# Compare raw model outputs (onnx vs dxnn)
python util/compare_raw_outputs.py \
    runs/predict/onnx/standalone/debug/raw_output/raw_output_[timestamp].npy \
    runs/predict/dxnn/standalone/debug/raw_output/raw_output_[timestamp].npy
```

### Checking ONNX Model Dynamic Shape Support

Utility to check whether an ONNX model supports dynamic shapes:

```bash
# Check default model
python util/check_onnx_dynamic.py

# Check specific model
python util/check_onnx_dynamic.py --model yolo11l/models/yolo11l.onnx
python util/check_onnx_dynamic.py -m yolo11l-seg/models/yolo11l-seg.onnx
```

**Example Output:**
```
================================================================================
ONNX Model Analysis: yolo11l/models/yolo11l.onnx
================================================================================

📥 Input Information:
--------------------------------------------------------------------------------

Input name: images
Shape:
  [0] Dynamic size: batch ⭐
  [1] Fixed size: 3
  [2] Dynamic size: height ⭐
  [3] Dynamic size: width ⭐
Full shape: ['batch', '3', 'height', 'width']

================================================================================
📤 Output Information:
--------------------------------------------------------------------------------

Output name: output0
Shape:
  [0] Dynamic size: batch ⭐
  [1] Fixed size: 84
  [2] Dynamic size: anchors ⭐
Full shape: ['batch', '84', 'anchors']

================================================================================
📊 Analysis Results:
--------------------------------------------------------------------------------
✅ This model supports dynamic shapes (dynamic=True).

Dynamic dimensions:
  - Input: Contains dynamic shape
  - Output: Contains dynamic shape

Available features:
  ✓ Batch processing available (BATCH_SIZE > 1)
  ✓ Various input sizes supported
  ✓ Runtime shape adjustment possible
================================================================================
```

**What is Dynamic Shape?**
- ✅ **Dynamic model**: Input size can be changed at runtime (e.g., 640x640, 480x640, 1024x1024)
- ❌ **Fixed model**: Input size is fixed (e.g., always 640x640 only)

**When to use:**
- Verify dynamic shape support after ONNX model export
- Check if rect=True mode is available (requires dynamic shape)
- Verify batch processing capability

### Debug Features

**standalone scripts:**
```python
# Enable debug mode in run_inference()
result_path = run_inference(MODEL_PATH, image_path, OUTPUT_DIR, debug=True)

# Outputs:
# - Preprocessed input tensor visualization
# - Raw model output (.npy file)
# - Detailed console logs
```

**Ultralytics DEEPX Scripts:**
```python
# Debug features automatically enabled via custom library
# Additional outputs:
# - Input tensor: debug/input/preprocessed_input_[timestamp].jpg
# - Raw output: debug/raw_output/raw_output_[timestamp].npy
# - Ultralytics output: debug/origin_output/
```


## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) - YOLOv11 model and framework
- [ONNX Runtime](https://onnxruntime.ai/) - Efficient inference engine
- [DEEPX](https://www.deepx.ai/) - Hardware acceleration runtime
- COCO Dataset - Training and evaluation dataset

**Built with ❤️ using YOLOv11, ONNX Runtime, and DEEPX**
