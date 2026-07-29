import sys
import json
from pathlib import Path

EXPORT_DIR = Path(__file__).parent
ONNX_PATH = EXPORT_DIR / "voice_nav_slm.onnx"
ONNX_INT8_PATH = EXPORT_DIR / "voice_nav_slm_int8.onnx"

def quantize_onnx_model():
    if not ONNX_PATH.exists():
        print(f"Error: ONNX model not found at {ONNX_PATH}. Run to_onnx.py first.")
        sys.exit(1)

    print(f"Quantizing ONNX model to INT8: {ONNX_PATH} -> {ONNX_INT8_PATH}...")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            model_input=str(ONNX_PATH),
            model_output=str(ONNX_INT8_PATH),
            weight_type=QuantType.QUInt8
        )

        orig_size = ONNX_PATH.stat().st_size / (1024 * 1024)
        quant_size = ONNX_INT8_PATH.stat().st_size / (1024 * 1024)

        print(f"Quantization complete!")
        print(f"  Original FP32 Size: {orig_size:.2f} MB")
        print(f"  Quantized INT8 Size: {quant_size:.2f} MB")
        print(f"  Compression Ratio:  {orig_size / quant_size:.2f}x")

    except ImportError:
        print("Note: 'onnxruntime' package with quantization module is not installed.")
        print("Install via: pip install onnxruntime")
    except Exception as e:
        print(f"Quantization error: {e}")

if __name__ == "__main__":
    quantize_onnx_model()
