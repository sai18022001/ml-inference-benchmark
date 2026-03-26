import urllib.request
import os
import sys
import hashlib

# ==============================================================
# MODEL DETAILS
# Source: Official ONNX Model Zoo (maintained by Microsoft)
# MobileNetV2 : image classification, 224x224 RGB input
# Output: 1000 class probabilities (ImageNet classes)
# ==============================================================
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenetv2.onnx")

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"[INFO] Model already exists: {MODEL_PATH}")
        print(f"[INFO] Size: {size_mb:.1f} MB")
        return MODEL_PATH

    print(f"[INFO] Downloading MobileNetV2 ONNX model...")
    print(f"[INFO] Source: ONNX Model Zoo (Microsoft)")
    print(f"[INFO] Destination: {MODEL_PATH}")
    print(f"[INFO] Please wait...\n")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            # \r = carriage return, overwrites same line
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, progress)
        print(f"\n\n[SUCCESS] Model downloaded successfully!")
        
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"[INFO] File size: {size_mb:.1f} MB")
        print(f"[INFO] Saved to: {MODEL_PATH}")
        return MODEL_PATH

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        sys.exit(1)

if __name__ == "__main__":
    path = download_model()
    print(f"\n[INFO] Ready for benchmarking: {path}")