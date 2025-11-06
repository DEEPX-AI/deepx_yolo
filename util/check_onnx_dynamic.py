#!/usr/bin/env python3
"""
Script to check whether an ONNX model supports dynamic shape

Usage:
    python check_onnx_dynamic.py
    python check_onnx_dynamic.py --model models/yolo11l.onnx
"""

import argparse
import onnx
from pathlib import Path


def check_onnx_dynamic(model_path):
    """
    Analyze ONNX model input/output shapes to check dynamic shape support.
    
    Args:
        model_path: Path to ONNX model file
    """
    print(f"\n{'='*80}")
    print(f"ONNX Model Analysis: {model_path}")
    print(f"{'='*80}\n")
    
    # Load ONNX model
    model = onnx.load(model_path)
    
    # Analyze input information
    print("📥 Input Information:")
    print("-" * 80)
    
    has_dynamic_input = False
    for input_tensor in model.graph.input:
        name = input_tensor.name
        shape = input_tensor.type.tensor_type.shape
        
        print(f"\nInput name: {name}")
        print(f"Shape:")
        
        dims = []
        for i, dim in enumerate(shape.dim):
            if dim.HasField('dim_value'):
                # Fixed size
                dim_value = dim.dim_value
                dims.append(str(dim_value))
                print(f"  [{i}] Fixed size: {dim_value}")
            elif dim.HasField('dim_param'):
                # Dynamic size
                dim_param = dim.dim_param
                dims.append(f"'{dim_param}'")
                has_dynamic_input = True
                print(f"  [{i}] Dynamic size: {dim_param} ⭐")
            else:
                dims.append("unknown")
                print(f"  [{i}] Unknown")
        
        print(f"Full shape: [{', '.join(dims)}]")
    
    # Analyze output information
    print(f"\n{'='*80}")
    print("📤 Output Information:")
    print("-" * 80)
    
    has_dynamic_output = False
    for output_tensor in model.graph.output:
        name = output_tensor.name
        shape = output_tensor.type.tensor_type.shape
        
        print(f"\nOutput name: {name}")
        print(f"Shape:")
        
        dims = []
        for i, dim in enumerate(shape.dim):
            if dim.HasField('dim_value'):
                # Fixed size
                dim_value = dim.dim_value
                dims.append(str(dim_value))
                print(f"  [{i}] Fixed size: {dim_value}")
            elif dim.HasField('dim_param'):
                # Dynamic size
                dim_param = dim.dim_param
                dims.append(f"'{dim_param}'")
                has_dynamic_output = True
                print(f"  [{i}] Dynamic size: {dim_param} ⭐")
            else:
                dims.append("unknown")
                print(f"  [{i}] Unknown")
        
        print(f"Full shape: [{', '.join(dims)}]")
    
    # Summary of results
    print(f"\n{'='*80}")
    print("📊 Analysis Results:")
    print("-" * 80)
    
    has_dynamic = has_dynamic_input or has_dynamic_output
    
    if has_dynamic:
        print("✅ This model supports dynamic shapes (dynamic=True).")
        print("\nDynamic dimensions:")
        if has_dynamic_input:
            print("  - Input: Contains dynamic shape")
        if has_dynamic_output:
            print("  - Output: Contains dynamic shape")
        print("\nAvailable features:")
        print("  ✓ Batch processing available (BATCH_SIZE > 1)")
        print("  ✓ Various input sizes supported")
        print("  ✓ Runtime shape adjustment possible")
    else:
        print("❌ This model uses fixed shapes (dynamic=False).")
        print("\nLimitations:")
        print("  ✗ Batch processing unavailable (BATCH_SIZE = 1 fixed)")
        print("  ✗ Only fixed input size supported")
        print("  ✗ Runtime shape changes not possible")
        print("\nHow to re-export:")
        print("  1. Change dynamic=True in export_onnx.py")
        print("  2. Run: python export_onnx.py")
    
    print(f"{'='*80}\n")
    
    return has_dynamic


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Check ONNX model dynamic shape support",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/yolo11l.onnx',
        help='Path to ONNX model file (default: models/yolo11l.onnx)'
    )
    
    args = parser.parse_args()
    
    # Set path based on current script location
    # script_dir = Path(__file__).parent
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        print(f"   Please check if the file exists.")
        return
    
    # Analyze model
    is_dynamic = check_onnx_dynamic(str(model_path))
    
    # Return exit code (can be used in scripts)
    exit(0 if is_dynamic else 1)


if __name__ == "__main__":
    main()
