# DEEPX-YOLO 프로젝트

이 프로젝트는 독립형 ONNX Runtime부터 DEEPX 가속 실행까지 다양한 YOLO 모델 추론 구현을 보여줍니다. 객체 감지, 포즈 추정, 인스턴스 분할, 방향성 경계 상자 감지 등 다양한 사용 사례를 위한 여러 예제 스크립트를 제공합니다.

## 🎯 프로젝트 개요

이 저장소는 다음을 포함합니다:
- **독립형 구현**: 외부 종속성 없음 (Ultralytics 없이 ONNX/DXNN 추론)
- **Ultralytics DEEPX 구현**: 향상된 디버깅 및 DXNN 지원을 위한 사용자 정의 Ultralytics DEEPX 라이브러리 사용
- **다중 작업 지원**: 감지, 포즈 추정, 분할, OBB
- **모델 변환 유틸리티**: PyTorch에서 ONNX로 내보내기 스크립트

## 🗂️ 디렉터리 구조

```plaintext
yolov11l_poc/
├── README.md
├── requirements.txt
├── assets/                     # 샘플 이미지
│   ├── boats.jpg
│   ├── bus.jpg
│   └── zidane.jpg
├── test_images/               # 테스트 이미지 폴더
│   └── 1.jpg ~ 7.jpg
├── lib/
│   └── ultralytics/           # 사용자 정의 Ultralytics DEEPX 라이브러리 (서브모듈)
├── yolo11l/                   # 객체 감지 예제
│   ├── export_onnx.py                              # PyTorch에서 ONNX로 변환
│   ├── ultralytics_deepx_lib_setup.py              # 사용자 정의 라이브러리 설정 스크립트
│   ├── predict_onnx_standalone.py                  # 독립형 ONNX 추론 (종속성 없음)
│   ├── predict_onnx_ultralytics_postprocess.py     # ONNX 추론 + Ultralytics 후처리 (하이브리드)
│   ├── predict_onnx_ultralytics_deepx.py           # 사용자 정의 Ultralytics DEEPX 라이브러리를 사용한 ONNX 추론
│   ├── predict_dxnn_standalone.py                  # 독립형 DXNN 추론 (종속성 없음)
│   ├── predict_dxnn_ultralytics_postprocess.py     # DXNN 추론 + Ultralytics 후처리 (하이브리드)
│   ├── predict_dxnn_ultralytics_deepx.py           # 사용자 정의 Ultralytics DEEPX 라이브러리를 사용한 DXNN 추론
│   ├── models/
│   │   ├── metadata.yaml
│   │   ├── yolo11l.pt                        # PyTorch 모델
│   │   ├── yolo11l.onnx                      # ONNX 모델
│   │   └── yolo11l.dxnn                      # DEEPX 모델
│   └── runs/predict/                         # 출력 결과
├── yolo11l-pose/              # 포즈 추정 예제
│   ├── ultralytics_deepx_lib_setup.py
│   ├── export_onnx.py
│   ├── predict_onnx_standalone.py
│   ├── predict_onnx_ultralytics_postprocess.py
│   ├── predict_onnx_ultralytics_deepx.py
│   ├── predict_dxnn_standalone.py
│   ├── predict_dxnn_ultralytics_postprocess.py
│   ├── predict_dxnn_ultralytics_deepx.py
│   └── models/
├── yolo11l-seg/               # 인스턴스 분할 예제
│   ├── ultralytics_deepx_lib_setup.py
│   ├── export_onnx.py
│   ├── predict_onnx_standalone.py
│   ├── predict_onnx_ultralytics_postprocess.py
│   ├── predict_onnx_ultralytics_deepx.py
│   ├── predict_dxnn_standalone.py
│   ├── predict_dxnn_ultralytics_postprocess.py
│   ├── predict_dxnn_ultralytics_deepx.py
│   └── models/
└── yolo11l-obb/               # 방향성 경계 상자 예제
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

## 🛠️ 사전 요구사항

### 1. Python 환경 요구사항

- Python 3.12 이상 (Python 3.12.3으로 테스트됨)

### 2. 필수 패키지 설치

```bash
# 가상 환경 생성 및 활성화 (권장)
python3 -m venv venv
source venv/bin/activate  # Linux
# venv\Scripts\activate   # Windows

# 필수 패키지 설치
pip install -r requirements.txt
```

### 3. 주요 종속성

**핵심 라이브러리:**
- **torch**: 텐서 연산을 위한 PyTorch
- **ultralytics**: YOLOv11 모델 로딩 및 변환 (Ultralytics DEEPX 예제용)
- **opencv-python**: 이미지 처리 및 시각화
- **numpy**: 수치 계산
- **onnxruntime**: ONNX 모델 추론

**DEEPX Runtime Python 라이브러리 (DXNN 추론용):**
- **dx-engine**: DXNN 모델 추론을 위한 DEEPX 런타임

### 4. 사용자 정의 Ultralytics DEEPX 라이브러리 설정

사용자 정의 Ultralytics DEEPX 라이브러리는 `lib/ultralytics/`에 Git 서브모듈로 포함되어 있습니다. 제공하는 기능:
- 입력 텐서의 디버그 시각화
- 원시 출력 텐서의 디버그 저장
- DXNN 모델 추론 지원

서브모듈을 초기화하려면:
```bash
git submodule update --init --recursive
```

`ultralytics_deepx_lib_setup.py` 스크립트는 이 사용자 정의 라이브러리를 사용하도록 Python 경로를 자동으로 구성합니다.

## 📥 모델 다운로드

### YOLOv11 모델 다운로드

YOLOv11 모델은 [Ultralytics 공식 문서](https://docs.ultralytics.com/models/yolo11/#performance-metrics)에서 다운로드할 수 있습니다.

```bash
# yolo11l 폴더로 이동
cd yolo11l/models

# YOLOv11l 모델 다운로드 (약 50MB)
# 방법 1: wget 사용
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt

# 방법 2: 직접 다운로드
# 브라우저에서 다운로드하여 yolo11l/models/ 폴더에 저장
```

**사용 가능한 모델:**
- `yolo11l.pt`: 객체 감지 (PyTorch 형식)
- `yolo11l-pose.pt`: 포즈 추정
- `yolo11l-seg.pt`: 인스턴스 분할
- `yolo11l-obb.pt`: 방향성 경계 상자 감지

## 🚀 사용법

### 1. 모델 변환 (PyTorch → ONNX)

```bash
cd yolo11l
python export_onnx.py
```

**export_onnx.py 기능:**
- `models/yolo11l.pt` → `models/yolo11l.onnx`로 변환
- 모델 구성과 함께 metadata.yaml 내보내기
- 최대 호환성을 위해 ONNX opset 21 사용
- 기업 환경을 위한 SSL 인증서 우회 지원

### 2. 객체 감지 추론

이 프로젝트는 **여섯 가지 다른 추론 구현**을 제공합니다:

#### 2.1. 독립형 ONNX 추론 (학습용으로 권장)

```bash
cd yolo11l
python predict_onnx_standalone.py
```

**기능:**
- ✅ **Ultralytics 종속성 없음** - 모든 함수가 단일 파일로 포팅됨
- ✅ **완전히 독립적** - 완전한 Boxes 및 Results 클래스 포함
- ✅ **교육용** - 전처리, 추론, 후처리를 이해하기 쉬움
- ✅ **재사용 가능** - 포즈, 분할, OBB 작업에 적용 가능

**구현 하이라이트:**
- 사용자 정의 전처리: letterbox, 정규화
- 직접 ONNX Runtime 실행
- 포팅된 NMS 및 좌표 스케일링 함수
- 모든 Boxes 속성을 가진 완전한 Results 객체

#### 2.2. Ultralytics 후처리를 사용한 ONNX 추론 (하이브리드)

```bash
cd yolo11l
python predict_onnx_ultralytics_postprocess.py
```

**기능:**
- ✅ **하이브리드 접근 방식** - 사용자 정의 전처리/추론 + Ultralytics 후처리
- ✅ **최소 종속성** - 후처리에만 Ultralytics 유틸리티 사용
- ✅ **유연성** - 전처리/추론은 사용자 정의 가능, 후처리는 검증됨
- ✅ **학습 친화적** - 각 단계를 명확하게 분리

**구현 하이라이트:**
- 사용자 정의 전처리: letterbox, 정규화
- 직접 ONNX Runtime 실행
- Ultralytics 후처리: non_max_suppression, ops.scale_boxes, Results 클래스
- YOLO 클래스 없이 Ultralytics 유틸리티만 사용

#### 2.3. Ultralytics DEEPX 라이브러리를 사용한 ONNX 추론

```bash
cd yolo11l
python predict_onnx_ultralytics_deepx.py
```

**기능:**
- 사용자 정의 Ultralytics DEEPX 라이브러리 사용
- 엔드투엔드 추론을 위한 완전한 YOLO 클래스
- 디버그 기능: 입력 텐서 시각화, 원시 출력 저장
- Ultralytics가 처리하는 모든 전처리/추론/후처리

#### 2.4. 독립형 DXNN 추론

```bash
cd yolo11l
python predict_dxnn_standalone.py
```

**기능:**
- 가속 추론을 위한 DEEPX 런타임
- Ultralytics 종속성 없음
- 독립형 ONNX 버전과 유사한 구조

#### 2.5. Ultralytics 후처리를 사용한 DXNN 추론 (하이브리드)

```bash
cd yolo11l
python predict_dxnn_ultralytics_postprocess.py
```

**기능:**
- ✅ **하이브리드 접근 방식** - 사용자 정의 전처리/DXNN 추론 + Ultralytics 후처리
- ✅ **DEEPX 가속** - DXNN 런타임을 통한 빠른 추론
- ✅ **검증된 후처리** - Ultralytics NMS 및 좌표 스케일링 사용
- ✅ **프로덕션 준비** - 성능과 정확도의 균형

**구현 하이라이트:**
- 사용자 정의 전처리: letterbox, 정규화
- DXNN Runtime 실행 (dx_engine)
- Ultralytics 후처리: non_max_suppression, ops.scale_boxes, Results 클래스
- YOLO 클래스 없이 DEEPX 가속 활용

#### 2.6. Ultralytics DEEPX 라이브러리를 사용한 DXNN 추론

```bash
cd yolo11l
python predict_dxnn_ultralytics_deepx.py
```

**기능:**
- 사용자 정의 Ultralytics 라이브러리를 통한 DEEPX 런타임 사용
- 완전한 YOLO 클래스 지원
- 향상된 디버깅 기능

### 3. 실행 프로세스

모든 추론 스크립트는 다음 패턴을 따릅니다:

1. **입력**: `../assets/` 또는 지정된 디렉터리에서 이미지 검색
2. **전처리**: Letterbox 크기 조정, 정규화, 채널 변환
3. **추론**: ONNX Runtime 또는 DEEPX 실행
4. **후처리**: NMS, 좌표 스케일링, Results 객체 생성
5. **시각화**: 경계 상자 그리기 및 결과 저장
6. **출력**: `runs/predict/{backend}/{script_name}/` 디렉터리에 저장

### 4. 실행 결과 예제

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

### 5. 비디오(Batch) 처리 실행 결과 예제 (predict_onnx_ultralytics_deepx_video.py)

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



## ⚙️ 구성 옵션

### 공통 구성 (모든 스크립트)

```python
# 모델 경로
MODEL_PATH = 'models/yolo11l.onnx'  # 또는 DXNN의 경우 'models/yolo11l.dxnn'
SOURCE_PATH = '../assets'            # 입력 이미지 경로 (파일 또는 디렉터리)
OUTPUT_DIR = 'runs/predict/...'     # 결과 저장 디렉터리

# 감지 매개변수 (Ultralytics 기본값)
CONFIDENCE_THRESHOLD = 0.25   # 신뢰도 임계값 (0.0 ~ 1.0)
IOU_THRESHOLD = 0.45         # NMS를 위한 IoU 임계값
INPUT_SIZE = 640             # 모델 입력 크기
```



## 🔧 문제 해결

### ONNX Runtime 오류

```bash
# CPU 버전 재설치
pip uninstall onnxruntime
pip install onnxruntime

# GPU 버전의 경우 (NVIDIA GPU 필요)
pip install onnxruntime-gpu
```

### 사용자 정의 Ultralytics 라이브러리를 찾을 수 없음

```bash
# Git 서브모듈 초기화
git submodule update --init --recursive

# lib/ultralytics/ 존재 확인
ls lib/ultralytics/

# ultralytics_deepx_lib_setup.py 스크립트가 경로 구성을 처리해야 함
```

### DEEPX Runtime 오류

```bash
# DXNN 추론을 위한 dx-engine 설치
pip install dx-engine

# 설치 확인
python -c "from dx_engine import InferenceEngine; print('DEEPX OK')"
```

## 📈 성능 정보

### 모델 사양

| 모델 | 크기 | mAP50-95 | 속도 (CPU) | 매개변수 | FLOPs |
|-------|------|----------|-------------|------------|-------|
| YOLOv11n | ~6MB | 39.5% | ~50ms | 2.6M | 6.5B |
| YOLOv11s | ~19MB | 47.0% | ~100ms | 9.4M | 21.5B |
| YOLOv11m | ~40MB | 51.5% | ~200ms | 20.1M | 68.0B |
| YOLOv11l | ~50MB | 53.4% | ~300ms | 25.3M | 86.9B |
| YOLOv11x | ~110MB | 54.7% | ~500ms | 56.9M | 194.9B |

### 추론 성능

**ONNX Runtime (CPU):**
- 이미지 전처리: ~10-20ms
- 모델 추론: ~200-500ms (모델 크기에 따라 다름)
- 후처리 (NMS): ~10-30ms
- 총: 이미지당 ~250-550ms

**DEEPX Runtime (가속):**
- 지원되는 하드웨어에서 상당한 속도 향상
- 최적화된 메모리 사용
- 배치 처리를 위한 낮은 지연 시간

### 지원되는 해상도

- 기본 입력: 640x640 (자동 letterbox 크기 조정)
- 최대 테스트: 1920x1080
- 최소 권장: 320x320

## 🎓 학습 자료

### 코드 이해하기

1. **독립형 스크립트로 시작**: 
   - `predict_onnx_standalone.py`가 최고의 시작점
   - 하나의 파일에 모든 전처리, 추론, 후처리 포함
   - 디버그 출력과 함께 잘 주석 처리됨

2. **이해해야 할 핵심 개념**:
   - **Letterbox 전처리**: 크기 조정 시 종횡비 유지
   - **NMS (Non-Maximum Suppression)**: 중복 감지 제거
   - **좌표 스케일링**: 모델 공간에서 이미지 공간으로 변환
   - **Results 객체**: 모든 감지 정보의 컨테이너

3. **진행 경로**:
   ```
   predict_onnx_standalone.py                  → 전체 파이프라인 이해
   predict_onnx_ultralytics_postprocess.py     → 하이브리드 접근 방식 학습
   predict_onnx_ultralytics_deepx.py           → Ultralytics DEEPX 통합 라이브러리 확인 (이미지)
   predict_onnx_ultralytics_deepx_video.py     → Ultralytics DEEPX 비디오 처리
   predict_dxnn_standalone.py                  → DEEPX 런타임 학습
   predict_dxnn_ultralytics_postprocess.py     → DEEPX + 검증된 후처리
   predict_dxnn_ultralytics_deepx.py           → Ultralytics DEEPX 통합 라이브러리 확인 (이미지)
   predict_dxnn_ultralytics_deepx_video.py     → Ultralytics DEEPX 비디오 처리 (DXNN 가속)
   ```

### 코드 재사용

standalone 스크립트에는 재사용 가능한 완전한 구현이 포함되어 있습니다:

```python
# predict_onnx_standalone.py에서

# 11개 속성을 가진 재사용 가능한 Boxes 클래스
class Boxes:
    - xyxy, xywh, xyxyn, xywhn  # 다양한 좌표 형식
    - conf, cls, id              # 감지 메타데이터
    - shape, is_track            # 속성
    - cpu(), numpy(), cuda()     # 장치 관리

# 재사용 가능한 유틸리티 함수
def letterbox()              # 종횡비 유지 크기 조정
def preprocess_image()       # 완전한 전처리 파이프라인
def non_max_suppression()    # NMS 구현
def scale_boxes()            # 좌표 변환
```

다음에 적용할 수 있습니다:
- 포즈 추정 (키포인트 처리 추가)
- 인스턴스 분할 (마스크 처리 추가)
- 방향성 경계 상자 (각도 처리 추가)

## 파일 설명

### 핵심 스크립트 (yolo11l/)

| 파일 | 목적 | 종속성 | 사용 사례 |
|------|---------|--------------|----------|
| `export_onnx.py` | PyTorch를 ONNX로 변환 | ultralytics, torch | 모델 준비 |
| `ultralytics_deepx_lib_setup.py` | 사용자 정의 라이브러리 경로 구성 | - | 라이브러리 초기화 |
| `predict_onnx_standalone.py` | ONNX 추론 (종속성 없음) | cv2, numpy, torch, onnxruntime | 학습, 사용자 정의 |
| `predict_onnx_ultralytics_postprocess.py` | ONNX 추론 + Ultralytics 후처리 | cv2, numpy, torch, onnxruntime, ultralytics (후처리만) | 하이브리드 접근, 검증된 후처리 |
| `predict_onnx_ultralytics_deepx.py` | ONNX 추론 (이미지, 전체 라이브러리) | ultralytics (사용자 정의), cv2, numpy, torch | 빠른 프로토타이핑, 디버깅 |
| `predict_onnx_ultralytics_deepx_video.py` | ONNX 추론 (비디오, 전체 라이브러리) | ultralytics (사용자 정의), cv2, numpy, torch | 비디오 처리, 프레임 단위 추론 |
| `predict_dxnn_standalone.py` | DXNN 추론 (종속성 없음) | cv2, numpy, torch, dx-engine | 프로덕션, 최소 종속성 |
| `predict_dxnn_ultralytics_postprocess.py` | DXNN 추론 + Ultralytics 후처리 | cv2, numpy, torch, dx-engine, ultralytics (후처리만) | DEEPX 가속 + 검증된 후처리 |
| `predict_dxnn_ultralytics_deepx.py` | DXNN 추론 (이미지, 전체 라이브러리) | ultralytics (사용자 정의), dx-engine | 가속화를 통한 개발 |
| `predict_dxnn_ultralytics_deepx_video.py` | DXNN 추론 (비디오, 전체 라이브러리) | ultralytics (사용자 정의), dx-engine | DXNN 가속 비디오 처리 |

### 출력 구조

```plaintext
yolo11l/runs/predict/
├── onnx/
│   ├── standalone/                    # predict_onnx_standalone.py 출력
│   │   ├── [input_image_name]_detected_[timestamp].jpg
│   │   └── debug/
│   │       ├── input/                 # 전처리된 입력 시각화
│   │       └── raw_output/            # 원시 모델 출력 (.npy)
│   ├── ultralytics_postprocess/       # predict_onnx_ultralytics_postprocess.py 출력
│   │   ├── [input_image_name]_detected_[timestamp].jpg
│   │   └── debug/
│   │       ├── input/                 # 전처리된 입력 시각화
│   │       └── raw_output/            # 원시 모델 출력 (.npy)
│   └── ultralytics_deepx/             # predict_onnx_ultralytics_deepx.py & _video.py 출력
│       ├── [input_image_name]_detected_[timestamp].jpg      # 이미지 출력
│       ├── [input_video_name]_detected_[timestamp].mp4      # 비디오 출력
│       └── debug/
│           ├── input/                 # 전처리된 입력 시각화
│           ├── raw_output/            # 원시 모델 출력 (.npy)
│           └── origin_output/         # Ultralytics 네이티브 출력
└── dxnn/
    ├── standalone/                    # predict_dxnn_standalone.py 출력
    │   ├── [input_image_name]_detected_[timestamp].jpg
    │   └── debug/
    │       ├── input/                 # 전처리된 입력 시각화
    │       └── raw_output/            # 원시 모델 출력 (.npy)
    ├── ultralytics_postprocess/       # predict_dxnn_ultralytics_postprocess.py 출력
    │   ├── [input_image_name]_detected_[timestamp].jpg
    │   └── debug/
    │       ├── input/                 # 전처리된 입력 시각화
    │       └── raw_output/            # 원시 모델 출력 (.npy)
    └── ultralytics_deepx/             # predict_dxnn_ultralytics_deepx.py & _video.py 출력
        ├── [input_image_name]_detected_[timestamp].jpg      # 이미지 출력
        ├── [input_video_name]_detected_[timestamp].mp4      # 비디오 출력
        └── debug/
            ├── input/                 # 전처리된 입력 시각화
            ├── raw_output/            # 원시 모델 출력 (.npy)
            └── origin_output/         # Ultralytics 네이티브 출력
    
```


## 🔍 디버깅 및 비교

### 출력 비교

프로젝트에는 서로 다른 구현 간의 출력을 비교하는 유틸리티가 포함되어 있습니다:

```bash
# 원시 모델 출력 비교(standalone vs Ultralytics DEEPX)
python util/compare_raw_outputs.py \
    runs/predict/onnx/standalone/debug/raw_output/raw_output_[timestamp].npy \
    runs/predict/onnx/ultralytics_deepx/debug/raw_output/raw_output_[timestamp].npy

# 원시 모델 출력 비교(onnx vs dxnn)
python util/compare_raw_outputs.py \
    runs/predict/onnx/standalone/debug/raw_output/raw_output_[timestamp].npy \
    runs/predict/dxnn/standalone/debug/raw_output/raw_output_[timestamp].npy
```

#### Tolerance(허용 오차) 옵션 가이드

`--tolerance` (또는 `-t`) 파라미터는 허용 가능한 차이의 임계값을 제어합니다:

```bash
# 매우 엄격한 비교 (FP32 vs FP32, 거의 완전히 동일해야 함)
python util/compare_raw_outputs.py file1.npy file2.npy --tolerance 1e-10

# 표준 비교 (부동소수점 오차 고려)
python util/compare_raw_outputs.py file1.npy file2.npy --tolerance 1e-6

# 양자화된 모델 비교 (INT8, 더 큰 오차 허용)
python util/compare_raw_outputs.py file1.npy file2.npy --tolerance 0.05
```

**권장 Tolerance 값:**

| 비교 대상 | Tolerance | 설명 |
|----------------|-----------|-------------|
| **FP32 vs FP32 (CPU vs GPU)** | `1e-10` ~ `1e-7` | 매우 엄격, 거의 동일해야 함 |
| **FP32 vs FP32 (다른 라이브러리)** | `1e-6` ~ `1e-5` | 표준, 부동소수점 오차 고려 |
| **FP32 vs FP16** | `1e-4` ~ `1e-3` | Mixed precision |
| **FP32 vs INT8 (NPU, 엄격)** | `0.10` (10%) | 중앙값 ~1-2%, 90분위수 ~8-12% |
| **FP32 vs INT8 (NPU, 표준)** | `0.15` (15%) | **권장** ⭐ 90% 값이 허용 범위 내 |
| **FP32 vs INT8 (NPU, 완화)** | `0.20` (20%) | 실용적 상한선, 95% 커버리지 |

**주의:** INT8 양자화 비교는 **백분위수 기반 검증**을 사용합니다 (90분위수가 허용 오차 내에 있어야 함). 이 방식은 모든 값을 일일이 검사하는 것보다 견고하며, 소수의 이상치는 허용하면서도 대부분의 출력이 정확함을 보장합니다.

**예제: ONNX vs DXNN 비교**

```bash
# 표준 15% 허용 오차 (NPU 양자화에 권장)
# 백분위수 기반 검증 사용: 90%의 값이 허용 오차 내에 있어야 함
python util/compare_raw_outputs.py \
    -f1 yolo11l/runs/predict/onnx/standalone/debug/raw_output/raw_output_*.npy \
    -f2 yolo11l/runs/predict/dxnn/standalone/debug/raw_output/raw_output_*.npy \
    -t 0.15
```

### ONNX 모델 동적 형상 확인

ONNX 모델이 동적 형상(dynamic shape)을 지원하는지 확인하는 유틸리티:

```bash
# 기본 모델 확인
python util/check_onnx_dynamic.py

# 특정 모델 확인
python util/check_onnx_dynamic.py --model yolo11l/models/yolo11l.onnx
python util/check_onnx_dynamic.py -m yolo11l-seg/models/yolo11l-seg.onnx
```

**출력 예시:**
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

**동적 형상(Dynamic Shape)이란?**
- ✅ **동적 모델**: 입력 크기를 런타임에 변경 가능 (예: 640x640, 480x640, 1024x1024)
- ❌ **고정 모델**: 입력 크기가 고정됨 (예: 항상 640x640만 가능)

**사용 시기:**
- ONNX 모델 export 후 동적 형상 지원 확인
- rect=True 모드 사용 가능 여부 확인 (동적 형상 필요)
- 배치 처리 가능 여부 확인

### 디버그 기능

**standalone 스크립트:**
```python
# run_inference()에서 디버그 모드 활성화
result_path = run_inference(MODEL_PATH, image_path, OUTPUT_DIR, debug=True)

# 출력:
# - 전처리된 입력 텐서 시각화
# - 원시 모델 출력 (.npy 파일)
# - 상세한 콘솔 로그
```

**Ultralytics DEEPX 스크립트:**
```python
# 사용자 정의 라이브러리를 통해 디버그 기능 자동 활성화
# 추가 출력:
# - 입력 텐서: debug/input/preprocessed_input_[timestamp].jpg
# - 원시 출력: debug/raw_output/raw_output_[timestamp].npy
# - Ultralytics 출력: debug/origin_output/
```


## 🙏 감사의 말

- [Ultralytics](https://ultralytics.com/) - YOLOv11 모델 및 프레임워크
- [ONNX Runtime](https://onnxruntime.ai/) - 효율적인 추론 엔진
- [DEEPX](https://www.deepx.ai/) - 하드웨어 가속 런타임
- COCO 데이터셋 - 학습 및 평가 데이터셋

**YOLOv11, ONNX Runtime, DEEPX를 사용하여 ❤️로 제작됨**
