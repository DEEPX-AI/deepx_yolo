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
├── yolo11l/                   # 객체 감지 예제
│   ├── export_onnx.py                        # PyTorch에서 ONNX로 변환
│   ├── predict_onnx_standalone.py            # 독립형 ONNX 추론 (종속성 없음)
│   ├── predict_dxnn_standalone.py            # 독립형 DXNN 추론 (종속성 없음)
│   ├── models/
│   │   ├── metadata.yaml
│   │   ├── yolo11l.pt                        # PyTorch 모델
│   │   ├── yolo11l.onnx                      # ONNX 모델
│   │   └── yolo11l.dxnn                      # DEEPX 모델
│   └── runs/predict/                         # 출력 결과
├── yolo11l-pose/              # 포즈 추정 예제
│   ├── export_onnx.py
│   ├── predict_onnx_standalone.py
│   ├── predict_dxnn_standalone.py
│   └── models/
├── yolo11l-seg/               # 인스턴스 분할 예제
│   ├── export_onnx.py
│   ├── predict_onnx_standalone.py
│   ├── predict_dxnn_standalone.py
│   └── models/
└── yolo11l-obb/               # 방향성 경계 상자 예제
│   ├── export_onnx.py
│   ├── predict_onnx_standalone.py
│   ├── predict_dxnn_standalone.py
    └── models/
```

## 파일 설명

### 핵심 스크립트 (yolo11l/)

| 파일 | 목적 | 종속성 | 사용 사례 |
|------|---------|--------------|----------|
| `export_onnx.py` | PyTorch를 ONNX로 변환 | ultralytics, torch | 모델 준비 |
| `predict_onnx_standalone.py` | ONNX 추론 (종속성 없음) | cv2, numpy, torch, onnxruntime | 학습, 사용자 정의 |
| `predict_dxnn_standalone.py` | DXNN 추론 (종속성 없음) | cv2, numpy, torch, dx-engine | 프로덕션, 최소 종속성 |

### 출력 구조

```plaintext
yolo11l/runs/predict/
├── onnx/
│   └── standalone/                    # predict_onnx_standalone.py 출력
│       ├── [input_image_name]_detected_[timestamp].jpg
│       └── debug/
│           ├── input/                 # 전처리된 입력 시각화
│           └── raw_output/            # 원시 모델 출력 (.npy)
└── dxnn/
    └── standalone/                    # predict_dxnn_standalone.py 출력
        ├── [input_image_name]_detected_[timestamp].jpg
        └── debug/
            ├── input/                 # 전처리된 입력 시각화
            └── raw_output/            # 원시 모델 출력 (.npy)
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

이 프로젝트는 **두 가지 다른 추론 구현**을 제공합니다:

#### 2.1. ONNX 추론
#### 2.1.1 Standalone ONNX 추론 (학습용으로 권장)

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

#### 2.2. DXNN 추론
##### 2.2.1. Standalone DXNN 추론

```bash
cd yolo11l
python predict_dxnn_standalone.py
```

**기능:**
- 가속 추론을 위한 DEEPX 런타임
- Ultralytics 종속성 없음
- Standalone ONNX 버전과 유사한 구조

### 3. 실행 프로세스

모든 추론 스크립트는 다음 패턴을 따릅니다:

1. **입력**: `../assets/` 또는 지정된 디렉터리에서 이미지 검색
2. **전처리**: Letterbox 크기 조정, 정규화, 채널 변환
3. **추론**: ONNX Runtime 또는 DEEPX 실행
4. **후처리**: NMS, 좌표 스케일링, Results 객체 생성
5. **시각화**: 경계 상자 그리기 및 결과 저장
6. **출력**: `runs/predict/{backend}/{script_name}/` 디렉터리에 저장


## 🔧 문제 해결

### ONNX Runtime 오류

```bash
# CPU 버전 재설치
pip uninstall onnxruntime
pip install onnxruntime

# GPU 버전의 경우 (NVIDIA GPU 필요)
pip install onnxruntime-gpu
```

### DEEPX Runtime 오류

```bash
# DXNN 추론을 위한 dx-engine 설치
pip install dx-engine

# 설치 확인
python -c "from dx_engine import InferenceEngine; print('DEEPX OK')"
```


## 🔍 디버깅 및 비교

### 출력 비교

프로젝트에는 서로 다른 구현 간의 출력을 비교하는 유틸리티가 포함되어 있습니다:

```bash
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

**상세 분포가 포함된 출력 예제:**

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

**출력 이해하기:**

- **Median relative diff (0.60%)**: 절반의 값이 0.6% 미만의 오차를 가짐 - 매우 우수!
- **90th percentile (8.74%)**: 90%의 값이 8.74% 이내의 오차 - 고품질 INT8 양자화
- **분포 히스토그램**: 대부분의 값(58%)이 1% 미만의 오차를 가진다는 것을 시각적으로 표현
- **주요 백분위수**: 50분위수부터 99분위수까지의 통계적 분석
- **백분위수 기반 검증**: 90분위수 ≤ tolerance(15%)이면 통과

이 상세한 출력을 통해 다음을 이해할 수 있습니다:
- 양자화가 얼마나 정확한지 (중앙값 ~0.6%)
- 각 오차 범위에 해당하는 값의 비율
- 이상치(outlier)가 어디에서 발생하는지 (90분위수 이상 10%)
- 전체 품질이 요구사항을 충족하는지 여부

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

## 🙏 감사의 말

- [Ultralytics](https://ultralytics.com/) - YOLOv11 모델 및 프레임워크
- [ONNX Runtime](https://onnxruntime.ai/) - 효율적인 추론 엔진
- [DEEPX](https://www.deepx.ai/) - 하드웨어 가속 런타임
- COCO 데이터셋 - 학습 및 평가 데이터셋

**YOLOv11, ONNX Runtime, DEEPX를 사용하여 ❤️로 제작됨**
