# Converts FP32 MobileNetV2 to INT8 using ONNX Runtime quantization
# this is Dynamic Quantization

from bz2 import compress
import os
import sys
from onnxruntime.quantization import quantize_dynamic, QuantType

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FP32     = os.path.join(PROJECT_ROOT, "models", "mobilenetv2.onnx")
MODEL_INT8     = os.path.join(PROJECT_ROOT, "models", "mobilenetv2_int8.onnx")

def quantize():
    if not os.path.exists(MODEL_FP32):
        print(f"[ERROR] FP32 model not found: {MODEL_FP32}")
        print("[INFO]  Run python/download_model.py first.")
        sys.exit(1)

    fp32_size = os.path.getsize(MODEL_FP32) / (1024 * 1024)
    print(f"\n[INFO] Original model (FP32): {fp32_size:.1f} MB")

    quantize_dynamic(model_input=MODEL_FP32,
                     model_output=MODEL_INT8,
                     weight_type=QuantType.QUInt8)

    int8_size = os.path.getsize(MODEL_INT8) / (1024 * 1024)
    compression = fp32_size / int8_size

    print(f"\n  FP32 model size : {fp32_size:.1f} MB")
    print(f"  INT8 model size : {int8_size:.1f} MB")
    print(f"  Compression     : {compression:.1f}x smaller")
    print("[INFO] Run compare_models.py to benchmark FP32 vs INT8")

if __name__ == "__main__":
    quantize()
