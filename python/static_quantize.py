# Performs static quantization on MobileNetV2

import os
import sys
import numpy as np
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FP32   = os.path.join(PROJECT_ROOT, "models", "mobilenetv2.onnx")
MODEL_STATIC = os.path.join(PROJECT_ROOT, "models", "mobilenetv2_static_int8.onnx")
CALIB_DIR    = os.path.join(PROJECT_ROOT, "models", "calibration_images")

SAMPLE_IMAGES = [
    "https://images.cocodataset.org/val2017/000000039769.jpg",
    "https://images.cocodataset.org/val2017/000000397133.jpg",
    "https://images.cocodataset.org/val2017/000000037777.jpg",
    "https://images.cocodataset.org/val2017/000000252219.jpg",
    "https://images.cocodataset.org/val2017/000000087038.jpg",
    "https://images.cocodataset.org/val2017/000000174482.jpg",
    "https://images.cocodataset.org/val2017/000000403385.jpg",
    "https://images.cocodataset.org/val2017/000000006818.jpg",
    "https://images.cocodataset.org/val2017/000000480985.jpg",
    "https://images.cocodataset.org/val2017/000000458054.jpg",
    "https://images.cocodataset.org/val2017/000000331352.jpg",
    "https://images.cocodataset.org/val2017/000000579321.jpg",
    "https://images.cocodataset.org/val2017/000000185250.jpg",
    "https://images.cocodataset.org/val2017/000000200365.jpg",
    "https://images.cocodataset.org/val2017/000000219578.jpg",
]

def download_calibration_images():
    os.makedirs(CALIB_DIR, exist_ok=True)

    # Check if already downloaded
    existing = [f for f in os.listdir(CALIB_DIR) if f.endswith(('.jpg','.png'))]
    if len(existing) >= len(SAMPLE_IMAGES):
        print(f"[INFO] Calibration images already present ({len(existing)} images)")
        return

    print(f"[INFO] Downloading {len(SAMPLE_IMAGES)} calibration images...")
    downloaded = 0
    for i, url in enumerate(SAMPLE_IMAGES):
        ext  = ".jpg" if ".jpg" in url else ".png"
        dest = os.path.join(CALIB_DIR, f"calib_{i:03d}{ext}")
        if os.path.exists(dest):
            downloaded += 1
            continue
        try:
            urllib.request.urlretrieve(url, dest)
            downloaded += 1
            sys.stdout.write(f"\r  Downloaded {downloaded}/{len(SAMPLE_IMAGES)}")
            sys.stdout.flush()
        except Exception as e:
            print(f"\n  [WARN] Failed to download image {i}: {e}")

    print(f"\n[INFO] {downloaded} calibration images ready in {CALIB_DIR}")

def build_calibration_data():
    try:
        from PIL import Image
    except ImportError:
        print("[INFO] Installing Pillow for image preprocessing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
        from PIL import Image

    images = []
    for fname in sorted(os.listdir(CALIB_DIR)):
        if not fname.endswith(('.jpg', '.png', '.jpeg')):
            continue
        path = os.path.join(CALIB_DIR, fname)
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((224, 224), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0

            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr  = (arr - mean) / std

            arr = arr.transpose(2, 0, 1)[np.newaxis, :]
            images.append(arr)
        except Exception as e:
            print(f"  [WARN] Skipping {fname}: {e}")

    print(f"[INFO] Prepared {len(images)} calibration tensors")
    return images

def run_static_quantization(calibration_tensors):
    from onnxruntime.quantization import (
        quantize_static,
        CalibrationDataReader,
        QuantType,
        QuantFormat
    )

    class MobileNetCalibReader(CalibrationDataReader):
        def __init__(self, tensors, input_name):
            self.tensors    = tensors
            self.input_name = input_name
            self.index      = 0

        def get_next(self):
            if self.index >= len(self.tensors):
                return None 
            data = {self.input_name: self.tensors[self.index]}
            self.index += 1
            return data

    import onnxruntime as ort
    sess = ort.InferenceSession(MODEL_FP32)
    input_name = sess.get_inputs()[0].name
    del sess

    print(f"[INFO] Running static quantization with {len(calibration_tensors)} images...")
    print(f"[INFO] This calibrates both weights AND activations...")

    calib_reader = MobileNetCalibReader(calibration_tensors, input_name)

    quantize_static(
        model_input=MODEL_FP32,
        model_output=MODEL_STATIC,
        calibration_data_reader=calib_reader,
        weight_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ, 
    )

    size_mb = os.path.getsize(MODEL_STATIC) / (1024 * 1024)
    print(f"[SUCCESS] Static INT8 model saved: {size_mb:.1f} MB")
    print(f"[INFO] Path: {MODEL_STATIC}")


def main():
    print("=" * 52)
    print("  Static Quantization: FP32 ? Static INT8")
    print("=" * 52)

    if not os.path.exists(MODEL_FP32):
        print("[ERROR] FP32 model not found. Run download_model.py first.")
        sys.exit(1)

    fp32_mb = os.path.getsize(MODEL_FP32) / (1024 * 1024)
    print(f"[INFO] Source model (FP32): {fp32_mb:.1f} MB")

    download_calibration_images()
    tensors = build_calibration_data()

    if len(tensors) == 0:
        print("[ERROR] No calibration images could be loaded.")
        sys.exit(1)

    run_static_quantization(tensors)

    static_mb = os.path.getsize(MODEL_STATIC) / (1024 * 1024)
    reduction = (1 - static_mb / fp32_mb) * 100
    print(f"\n  FP32 size   : {fp32_mb:.1f} MB")
    print(f"  Static INT8 : {static_mb:.1f} MB")
    print(f"  Reduction   : {reduction:.0f}%")
    print("\n[INFO] Run compare_all_models.py to benchmark all three")

if __name__ == "__main__":
    main()