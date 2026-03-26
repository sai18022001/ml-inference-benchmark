
# py python/run_benchmark.py
# py python/run_benchmark.py --iterations 100


import subprocess
import sys
import os
import json
import argparse
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "mobilenetv2.onnx")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
RESULTS_JSON = os.path.join(RESULTS_DIR, "benchmark_results.json")

if sys.platform == "win32":
    BINARY_PATHS = [
        os.path.join(PROJECT_ROOT, "bin", "ml_benchmark.exe"),
        os.path.join(PROJECT_ROOT, "out", "build", "x64-Release",
                     "ml_benchmark.exe"),
    ]
else:
    BINARY_PATHS = [
        os.path.join(PROJECT_ROOT, "bin", "ml_benchmark"),
        os.path.join(PROJECT_ROOT, "build", "ml_benchmark"),
    ]

def find_binary():
    for path in BINARY_PATHS:
        if os.path.exists(path):
            return path
    return None

def print_header():
    print("=" * 56)
    print("   ML Inference Benchmarking Tool : Full Pipeline")
    print("   Model: MobileNetV2 ; Runtime: ONNX Runtime")
    print("=" * 56)


def ensure_model():
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"[1/3] Model ready: {size_mb:.1f} MB")
        return True

    print("[1/3] Model not found. Downloading...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(PROJECT_ROOT, "python", "download_model.py")],
        cwd=PROJECT_ROOT
    )
    return result.returncode == 0

# Run C++ Benchmark
def run_benchmark():
    binary = find_binary()
    if not binary:
        print("[ERROR] Binary not found. Please build the project first.")
        print("        In VS2022: Build then Build All (Ctrl+Shift+B)")
        return False

    print(f"[2/3] Running C++ benchmark binary...")
    print(f"      Binary: {os.path.relpath(binary, PROJECT_ROOT)}")
    print()

    start = time.time()
    result = subprocess.run(
        [binary, MODEL_PATH, RESULTS_DIR],
        cwd=PROJECT_ROOT
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"[ERROR] Benchmark failed with code {result.returncode}")
        return False

    print(f"\n[2/3] Benchmark complete in {elapsed:.1f}s")
    return True

# Generate Charts
def run_visualizer():
    print(f"\n[3/3] Generating charts...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(PROJECT_ROOT, "python", "visualize.py")],
        cwd=PROJECT_ROOT
    )
    return result.returncode == 0

# Prints Summary
def print_summary():
    if not os.path.exists(RESULTS_JSON):
        return

    with open(RESULTS_JSON) as f:
        results = json.load(f)

    print("\n" + "=" * 56)
    print("  BENCHMARK SUMMARY REPORT")
    print("  Model: MobileNetV2 ; Platform: CPU")
    print("=" * 56)
    print(f"  {'Config':<20} {'Mean(ms)':>10} {'P99(ms)':>10} {'FPS':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    for r in results:
        config = f"B{r['batch_size']} T{r['num_threads']}"
        print(f"  {config:<20} "
              f"{r['latency_mean_ms']:>10.2f} "
              f"{r['latency_p99_ms']:>10.2f} "
              f"{r['throughput_fps']:>10.1f}")

    best_fps = max(results, key=lambda x: x["throughput_fps"])
    best_lat = min(results, key=lambda x: x["latency_mean_ms"])

    print(f"\n  Best throughput : "
          f"Batch={best_fps['batch_size']} Threads={best_fps['num_threads']} "
          f"-> {best_fps['throughput_fps']:.1f} FPS")
    print(f"  Best latency    : "
          f"Batch={best_lat['batch_size']} Threads={best_lat['num_threads']} "
          f"-> {best_lat['latency_mean_ms']:.2f} ms")
    print(f"\n  Charts saved to: results/")
    print(f"    throughput_heatmap.png")
    print(f"    latency_bars.png")
    print(f"    throughput_scaling.png")
    print(f"    latency_vs_throughput.png")
    print("=" * 56)


def main():
    print_header()

    if not ensure_model():
        print("[ERROR] Could not get model. Exiting.")
        sys.exit(1)

    if not run_benchmark():
        print("[ERROR] Benchmark failed. Exiting.")
        sys.exit(1)

    if not run_visualizer():
        print("[ERROR] Visualization failed. Exiting.")
        sys.exit(1)

    print_summary()
    print("\n[SUCCESS] Full pipeline complete!")

if __name__ == "__main__":
    main()