#!/usr/bin/env python3
"""
ONNX 모델의 동적 형상(dynamic shape) 지원 여부를 확인하는 스크립트

사용법:
    python check_onnx_dynamic.py
    python check_onnx_dynamic.py --model models/yolo11l.onnx
"""

import argparse
import onnx
from pathlib import Path


def check_onnx_dynamic(model_path):
    """
    ONNX 모델의 입출력 형상을 분석하여 동적 형상 지원 여부를 확인합니다.
    
    Args:
        model_path: ONNX 모델 파일 경로
    """
    print(f"\n{'='*80}")
    print(f"ONNX 모델 분석: {model_path}")
    print(f"{'='*80}\n")
    
    # ONNX 모델 로드
    model = onnx.load(model_path)
    
    # 입력 정보 분석
    print("📥 입력(Input) 정보:")
    print("-" * 80)
    
    has_dynamic_input = False
    for input_tensor in model.graph.input:
        name = input_tensor.name
        shape = input_tensor.type.tensor_type.shape
        
        print(f"\n입력 이름: {name}")
        print(f"형상:")
        
        dims = []
        for i, dim in enumerate(shape.dim):
            if dim.HasField('dim_value'):
                # 고정 크기
                dim_value = dim.dim_value
                dims.append(str(dim_value))
                print(f"  [{i}] 고정 크기: {dim_value}")
            elif dim.HasField('dim_param'):
                # 동적 크기
                dim_param = dim.dim_param
                dims.append(f"'{dim_param}'")
                has_dynamic_input = True
                print(f"  [{i}] 동적 크기: {dim_param} ⭐")
            else:
                dims.append("unknown")
                print(f"  [{i}] 알 수 없음")
        
        print(f"전체 형상: [{', '.join(dims)}]")
    
    # 출력 정보 분석
    print(f"\n{'='*80}")
    print("📤 출력(Output) 정보:")
    print("-" * 80)
    
    has_dynamic_output = False
    for output_tensor in model.graph.output:
        name = output_tensor.name
        shape = output_tensor.type.tensor_type.shape
        
        print(f"\n출력 이름: {name}")
        print(f"형상:")
        
        dims = []
        for i, dim in enumerate(shape.dim):
            if dim.HasField('dim_value'):
                # 고정 크기
                dim_value = dim.dim_value
                dims.append(str(dim_value))
                print(f"  [{i}] 고정 크기: {dim_value}")
            elif dim.HasField('dim_param'):
                # 동적 크기
                dim_param = dim.dim_param
                dims.append(f"'{dim_param}'")
                has_dynamic_output = True
                print(f"  [{i}] 동적 크기: {dim_param} ⭐")
            else:
                dims.append("unknown")
                print(f"  [{i}] 알 수 없음")
        
        print(f"전체 형상: [{', '.join(dims)}]")
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("📊 분석 결과:")
    print("-" * 80)
    
    has_dynamic = has_dynamic_input or has_dynamic_output
    
    if has_dynamic:
        print("✅ 이 모델은 동적 형상(dynamic=True)을 지원합니다.")
        print("\n동적 차원:")
        if has_dynamic_input:
            print("  - 입력: 동적 형상 포함")
        if has_dynamic_output:
            print("  - 출력: 동적 형상 포함")
        print("\n사용 가능한 기능:")
        print("  ✓ 배치 처리 가능 (BATCH_SIZE > 1)")
        print("  ✓ 다양한 입력 크기 처리 가능")
        print("  ✓ 런타임에 형상 조정 가능")
    else:
        print("❌ 이 모델은 고정 형상(dynamic=False)을 사용합니다.")
        print("\n제약사항:")
        print("  ✗ 배치 처리 불가 (BATCH_SIZE = 1 고정)")
        print("  ✗ 고정된 입력 크기만 처리 가능")
        print("  ✗ 런타임 형상 변경 불가")
        print("\n재 export 방법:")
        print("  1. export_onnx.py에서 dynamic=True로 변경")
        print("  2. python export_onnx.py 실행")
    
    print(f"{'='*80}\n")
    
    return has_dynamic


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="ONNX 모델의 동적 형상 지원 여부 확인",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/yolo11l.onnx',
        help='ONNX 모델 파일 경로 (기본값: models/yolo11l.onnx)'
    )
    
    args = parser.parse_args()
    
    # 현재 스크립트 위치 기준으로 경로 설정
    # script_dir = Path(__file__).parent
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        print(f"   파일이 존재하는지 확인해주세요.")
        return
    
    # 모델 분석
    is_dynamic = check_onnx_dynamic(str(model_path))
    
    # 종료 코드 반환 (스크립트에서 활용 가능)
    exit(0 if is_dynamic else 1)


if __name__ == "__main__":
    main()
