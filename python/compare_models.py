# Benchmarks FP32 vs INT8 model side by side.
# Shows latency improvement and file size reduction.

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import onnxruntime as ort

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FP32   = os.path.join(PROJECT_ROOT, "models", "mobilenetv2.onnx")
MODEL_INT8   = os.path.join(PROJECT_ROOT, "models", "mobilenetv2_int8.onnx")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

def benchmark_model(model_path, label, iterations=50, warmup=5):
    print(f"\n[BENCH] Benchmarking {label}...")
    print(f"        Model: {os.path.basename(model_path)}")

    # Create session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, opts)

    # Get input shape from model
    input_name  = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    # Replace dynamic dim (-1 or None) with 1
    shape = [1 if (d is None or d == -1 or isinstance(d, str)) else d
             for d in input_shape]

    # Dummy input
    dummy = np.full(shape, 0.5, dtype=np.float32)

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    # Measured runs
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    latencies.sort()
    mean = sum(latencies) / len(latencies)
    p50  = latencies[int(0.50 * len(latencies))]
    p95  = latencies[int(0.95 * len(latencies))]
    p99  = latencies[int(0.99 * len(latencies))]
    fps  = 1000.0 / mean

    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    result = {
        "label":    label,
        "size_mb":  round(size_mb, 2),
        "mean_ms":  round(mean, 3),
        "p50_ms":   round(p50,  3),
        "p95_ms":   round(p95,  3),
        "p99_ms":   round(p99,  3),
        "fps":      round(fps,  1),
    }

    print(f"        Mean: {mean:.2f}ms | P99: {p99:.2f}ms | "
          f"{fps:.0f} FPS | Size: {size_mb:.1f}MB")
    return result

def plot_comparison(fp32, int8, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("FP32 vs INT8 Quantization : MobileNetV2",
                 fontsize=14, fontweight="bold", y=1.02)

    colors = {"FP32": "#2563EB", "INT8": "#16A34A"}
    models = [fp32["label"], int8["label"]]
    clrs   = [colors["FP32"], colors["INT8"]]

    # --- Chart 1: Model Size ---
    ax = axes[0]
    sizes = [fp32["size_mb"], int8["size_mb"]]
    bars = ax.bar(models, sizes, color=clrs, width=0.45, edgecolor="white")
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f} MB", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Model Size", fontweight="bold")
    ax.set_ylabel("Size (MB)")
    ax.set_ylim(0, fp32["size_mb"] * 1.3)
    ax.grid(axis="y", alpha=0.3)
    reduction = (1 - int8["size_mb"] / fp32["size_mb"]) * 100
    ax.text(0.5, 0.85, f"{reduction:.0f}% smaller",
            transform=ax.transAxes, ha="center",
            color="#16A34A", fontsize=10, fontweight="bold")

    # --- Chart 2: Latency Comparison ---
    ax = axes[1]
    x = np.arange(3)
    width = 0.32
    fp32_vals = [fp32["mean_ms"], fp32["p95_ms"], fp32["p99_ms"]]
    int8_vals = [int8["mean_ms"], int8["p95_ms"], int8["p99_ms"]]
    b1 = ax.bar(x - width/2, fp32_vals, width, label="FP32",
                color=colors["FP32"], alpha=0.9)
    b2 = ax.bar(x + width/2, int8_vals, width, label="INT8",
                color=colors["INT8"], alpha=0.9)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Mean", "P95", "P99"])
    ax.set_title("Latency (ms) : Lower is better", fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    speedup = fp32["mean_ms"] / int8["mean_ms"]
    ax.text(0.5, 0.9, f"{speedup:.2f}x speedup",
            transform=ax.transAxes, ha="center",
            color="#16A34A", fontsize=10, fontweight="bold")

    # --- Chart 3: Throughput ---
    ax = axes[2]
    fps_vals = [fp32["fps"], int8["fps"]]
    bars = ax.bar(models, fps_vals, color=clrs, width=0.45, edgecolor="white")
    for bar, val in zip(bars, fps_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f} FPS", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_title("Throughput : Higher is better", fontweight="bold")
    ax.set_ylabel("Inferences / second (FPS)")
    ax.set_ylim(0, max(fps_vals) * 1.3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "quantization_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[INFO] Chart saved: {path}")
    return path

def print_summary(fp32, int8):
    speedup    = fp32["mean_ms"] / int8["mean_ms"]
    size_reduc = (1 - int8["size_mb"] / fp32["size_mb"]) * 100

    print("\n" + "=" * 52)
    print("  QUANTIZATION COMPARISON REPORT")
    print("=" * 52)
    print(f"  {'Metric':<22} {'FP32':>10} {'INT8':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    print(f"  {'Model size (MB)':<22} {fp32['size_mb']:>10.1f} {int8['size_mb']:>10.1f}")
    print(f"  {'Mean latency (ms)':<22} {fp32['mean_ms']:>10.2f} {int8['mean_ms']:>10.2f}")
    print(f"  {'P95 latency (ms)':<22} {fp32['p95_ms']:>10.2f} {int8['p95_ms']:>10.2f}")
    print(f"  {'P99 latency (ms)':<22} {fp32['p99_ms']:>10.2f} {int8['p99_ms']:>10.2f}")
    print(f"  {'Throughput (FPS)':<22} {fp32['fps']:>10.1f} {int8['fps']:>10.1f}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    print(f"  {'Speedup':<22} {'':>10} {speedup:>9.2f}x")
    print(f"  {'Size reduction':<22} {'':>10} {size_reduc:>9.0f}%")
    print("=" * 52)

def main():
    print("=" * 52)
    print("  FP32 vs Dynamic INT8 vs Static INT8")
    print("  Model: MobileNetV2 | Threads: 4")
    print("=" * 52)

    MODEL_STATIC = os.path.join(PROJECT_ROOT, "models", "mobilenetv2_static_int8.onnx")

    models_to_run = [(MODEL_FP32, "FP32"), (MODEL_INT8, "Dynamic INT8")]
    if os.path.exists(MODEL_STATIC):
        models_to_run.append((MODEL_STATIC, "Static INT8"))
    else:
        print("[WARN] Static INT8 model not found : run static_quantize.py first")

    for path, label in models_to_run:
        if not os.path.exists(path):
            print(f"[ERROR] {label} model not found: {path}")
            sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = [benchmark_model(path, label) for path, label in models_to_run]

    # Print summary
    fp32 = results[0]
    print("\n" + "=" * 60)
    print("  QUANTIZATION COMPARISON REPORT")
    print("=" * 60)
    print(f"  {'Metric':<22} {'FP32':>12} {'Dyn INT8':>12} {'Stat INT8':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12}")
    metrics = [
        ("Model size (MB)",    "size_mb",  ".1f"),
        ("Mean latency (ms)",  "mean_ms",  ".2f"),
        ("P99 latency (ms)",   "p99_ms",   ".2f"),
        ("Throughput (FPS)",   "fps",      ".1f"),
    ]
    for label, key, fmt in metrics:
        row = f"  {label:<22}"
        for r in results:
            row += f" {r[key]:>12{fmt}}"
        print(row)
    print("=" * 60)

    # Save JSON
    out = {r["label"]: r for r in results}
    json_path = os.path.join(RESULTS_DIR, "quantization_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    # Generate chart with all three models
    if len(results) >= 2:
        plot_three_way(results, RESULTS_DIR)

    print("\n[SUCCESS] Three-way comparison complete!")


def plot_three_way(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("FP32 vs Dynamic INT8 vs Static INT8 : MobileNetV2",
                 fontsize=13, fontweight="bold", y=1.02)

    palette = ["#2563EB", "#16A34A", "#D97706"]
    labels  = [r["label"] for r in results]
    colors  = palette[:len(results)]

    # Size
    ax = axes[0]
    bars = ax.bar(labels, [r["size_mb"] for r in results],
                  color=colors, width=0.5, edgecolor="white")
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{r['size_mb']:.1f}MB", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Model Size (MB)", fontweight="bold")
    ax.set_ylabel("MB")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis='x', labelsize=8)

    # Latency
    ax = axes[1]
    x = np.arange(3)
    w = 0.25
    for i, r in enumerate(results):
        vals = [r["mean_ms"], r["p95_ms"], r["p99_ms"]]
        bars = ax.bar(x + (i - len(results)/2 + 0.5) * w, vals,
                      w, label=r["label"], color=colors[i], alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                    f"{h:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Mean", "P95", "P99"])
    ax.set_title("Latency (ms)", fontweight="bold")
    ax.set_ylabel("ms")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Throughput
    ax = axes[2]
    bars = ax.bar(labels, [r["fps"] for r in results],
                  color=colors, width=0.5, edgecolor="white")
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{r['fps']:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Throughput (FPS)", fontweight="bold")
    ax.set_ylabel("FPS")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "quantization_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Chart saved: {path}")

if __name__ == "__main__":
    main()