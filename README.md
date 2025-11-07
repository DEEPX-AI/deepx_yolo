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
├── assets/                       # Sample images
│   ├── boats.jpg
│   ├── bus.jpg
│   └── zidane.jpg
├── test_images/                  # Test images folder
│   └── 1.jpg ~ 7.jpg
├── lib/
│   └── ultralytics/              # Custom Ultralytics DEEPX library (submodule)
├── utils/                        # Utility scripts
│   ├── check_onnx_dynamic.py     # check dynamic input mode for ONNX Model
│   └── compare_onnx_outputs.py   # compare ONNX outputs
├── yolo11l/                      # Object Detection examples
│   ├── export_onnx.py                            # PyTorch to ONNX conversion
│   ├── ultralytics_deepx_lib_setup.py            # Custom library setup script
│   ├── predict_onnx_standalone.py                # Standalone ONNX inference (zero dependencies)
│   ├── predict_onnx_ultralytics_postprocess.py   # ONNX inference + Ultralytics postprocessing (hybrid)
│   ├── predict_onnx_ultralytics_deepx.py         # ONNX inference with custom Ultralytics DEEPX library
│   ├── predict_onnx_ultralytics_deepx_video.py   # Video(Batch) ONNX inference with custom Ultralytics DEEPX library
│   ├── predict_dxnn_standalone.py                # Standalone DXNN inference (zero dependencies)
│   ├── predict_dxnn_ultralytics_postprocess.py   # DXNN inference + Ultralytics postprocessing (hybrid)
│   ├── predict_dxnn_ultralytics_deepx.py         # DXNN inference with custom Ultralytics DEEPX library
│   ├── predict_dxnn_ultralytics_deepx_video.py   # Video(Batch) DXNN inference with custom Ultralytics DEEPX library
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

The project provides **eight different inference implementations**:

#### 2.1. ONNX Inference
##### 2.1.1. Standalone ONNX Inference (Recommended for learning)

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

##### 2.1.2. ONNX Inference with Ultralytics Postprocessing (Hybrid)

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

##### 2.1.3. ONNX Inference with Ultralytics DEEPX Library

```bash
cd yolo11l
python predict_onnx_ultralytics_deepx.py
```

**Features:**
- Uses custom Ultralytics DEEPX library
- Complete YOLO class for end-to-end inference
- Debug features: input tensor visualization, raw output saving
- All preprocessing/inference/postprocessing handled by Ultralytics

##### 2.1.4. Video(Batch) ONNX Inference with Ultralytics DEEPX Library

```bash
cd yolo11l
python predict_onnx_ultralytics_deepx_video.py
```

**Features:**
- Uses custom Ultralytics DEEPX library
- Complete YOLO class for end-to-end inference
- Debug features: input tensor visualization, raw output saving
- All preprocessing/inference/postprocessing handled by Ultralytics

#### 2.2. DXNN Inference
##### 2.2.1. Standalone DXNN Inference

```bash
cd yolo11l
python predict_dxnn_standalone.py
```

**Features:**
- DEEPX runtime for accelerated inference
- Zero Ultralytics dependencies
- Similar structure to standalone ONNX version

##### 2.2.2. DXNN Inference with Ultralytics Postprocessing (Hybrid)

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

##### 2.2.3. DXNN Inference with Ultralytics DEEPX Library

```bash
cd yolo11l
python predict_dxnn_ultralytics_deepx.py
```

**Features:**
- Uses DEEPX runtime via custom Ultralytics library
- Complete YOLO class support
- Enhanced debugging capabilities

##### 2.2.4. Video(Batch) DXNN Inference with Ultralytics DEEPX Library

```bash
cd yolo11l
python predict_dxnn_ultralytics_deepx_video.py
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


## 🔧 Troubleshooting

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

#### Tolerance Option Guide

The `--tolerance` (or `-t`) parameter controls the acceptable difference threshold:

```bash
# Very strict comparison (FP32 vs FP32, almost identical required)
python util/compare_raw_outputs.py file1.npy file2.npy --tolerance 1e-10

# Standard comparison (floating-point error considered)
python util/compare_raw_outputs.py file1.npy file2.npy --tolerance 1e-6

# Quantized model comparison (INT8, larger error allowed)
python util/compare_raw_outputs.py file1.npy file2.npy --tolerance 0.05
```

**Recommended Tolerance Values:**

| Comparison Type | Tolerance | Description |
|----------------|-----------|-------------|
| **FP32 vs FP32 (CPU vs GPU)** | `1e-10` ~ `1e-7` | Very strict, almost identical |
| **FP32 vs FP32 (different libs)** | `1e-6` ~ `1e-5` | Standard, floating-point error considered |
| **FP32 vs FP16** | `1e-4` ~ `1e-3` | Mixed precision |
| **FP32 vs INT8 (NPU, strict)** | `0.10` (10%) | Median ~1-2%, 90th percentile ~8-12% |
| **FP32 vs INT8 (NPU, standard)** | `0.15` (15%) | **Recommended** ⭐ 90% values within tolerance |
| **FP32 vs INT8 (NPU, relaxed)** | `0.20` (20%) | Practical upper limit, 95% coverage |

**Note:** INT8 quantization comparison uses **percentile-based validation** (90th percentile must be within tolerance). This is more robust than checking every single value, as it allows for a small percentage of outliers while ensuring the majority of outputs are accurate.

**Example: ONNX vs DXNN Comparison**

```bash
# Standard 15% tolerance (recommended for NPU quantization)
# Uses percentile-based validation: 90% of values must be within tolerance
python util/compare_raw_outputs.py \
    -f1 yolo11l/runs/predict/onnx/standalone/debug/raw_output/raw_output_*.npy \
    -f2 yolo11l/runs/predict/dxnn/standalone/debug/raw_output/raw_output_*.npy \
    -t 0.15
```

**Example Output with Detailed Distribution:**

```
🚀 Starting Raw Output Comparison...
📁 Comparing:
   File 1: raw_output0_zidane_20251106_113527_583.npy
   File 2: raw_output0_zidane_20251105_205047_003.npy
================================================================================
🔍 RAW OUTPUT COMPARISON RESULTS
================================================================================

📁 Files:
   File 1: raw_output0_zidane_20251106_113527_583.npy
   File 2: raw_output0_zidane_20251105_205047_003.npy

📊 Basic Info:
   Shape 1: (1, 56, 8400)
   Shape 2: (1, 56, 8400)
   Shape Match: ✅
   Data Type 1: float32
   Data Type 2: float32
   Total Elements: 470,400

📈 Statistics:
   File 1 - Min: -20.210140, Max: 785.002502
   File 1 - Mean: 28.104315, Std: 84.515289
   File 2 - Min: 0.000000, Max: 796.703125
   File 2 - Mean: 28.123386, Std: 84.544266

🔍 Comparison:
   Exact Equal: ❌
   Tolerance: 0.15 = 15.0% relative error
   Median relative diff: 0.60%
   90th percentile diff: 8.74%
   Within tolerance: 95.5% of non-zero values
   Status (90th percentile ≤ 15.0% relative error): ✅
   Max Absolute Difference: 191.7122802734
   Mean Absolute Difference: 1.0487236977
   Max Relative Difference: 99.9657 (9996.57%)
   Mean Relative Difference: 0.0326 (3.26%)
   Different Elements: 275,506 (58.5685%)

🎯 FINAL RESULT:
   ✅ NEARLY_IDENTICAL (within tolerance 0.15)
   📝 The arrays are numerically equivalent within tolerance.
================================================================================

📊 DETAILED ERROR DISTRIBUTION:
--------------------------------------------------------------------------------

   Total non-zero values: 462,248
   Near-zero values (<1e-5): 8,152 (1.73%)

   📈 Relative Error Distribution (non-zero values only):
   Range                     Count  Percent  Cumulative Visualization
   -------------------- ---------- -------- ----------- --------------
   0.0% - 1.0%             269,529   58.31%       58.3% █████████████████
   1.0% - 5.0%             111,004   24.01%       82.3% ███████
   5.0% - 10.0%             42,777    9.25%       91.6% ██
   10.0% - 15.0%            17,979    3.89%       95.5% █
   15.0% - 20.0%             8,661    1.87%       97.3% 
   20.0% - 50.0%            11,034    2.39%       99.7% 
   50.0%+                    1,264    0.27%      100.0% 

   📍 Key Percentiles:
      50th percentile:   0.60%
      75th percentile:   3.05%
      80th percentile:   4.28%
      85th percentile:   6.02%
      90th percentile:   8.74% ⭐
      95th percentile:  14.17%
      99th percentile:  30.83%

   🎯 90th Percentile Analysis:
      90% of values: 0.00% ~ 8.74%
      10% of values: 8.74% ~ 9996.57%
      Tolerance threshold: 15.0%
      Result: ✅ PASS (90th percentile 8.74% ≤ 15.0%)
```

**Understanding the Output:**

- **Median relative diff (0.60%)**: Half of the values have less than 0.6% error - excellent!
- **90th percentile (8.74%)**: 90% of values are within 8.74% error - high quality INT8 quantization
- **Distribution histogram**: Visual representation showing most values (58%) have <1% error
- **Key percentiles**: Statistical breakdown from 50th to 99th percentile
- **Percentile-based validation**: Passes if 90th percentile ≤ tolerance (15%)

This detailed output helps you understand:
- How accurate the quantization is (median ~0.6%)
- What percentage of values fall within each error range
- Where outliers occur (10% above 90th percentile)
- Whether the overall quality meets your requirements

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
