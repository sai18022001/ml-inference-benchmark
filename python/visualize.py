# Reads benchmark_results.json and generates professional charts for the benchmark report.


import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "benchmark_results.json"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

COLORS = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]

def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"[INFO] Loaded {len(df)} benchmark results")
    print(df[["batch_size", "num_threads",
              "latency_mean_ms", "throughput_fps"]].to_string(index=False))
    return df


# CHART 1: Throughput Heatmap

def plot_throughput_heatmap(df, output_dir):
    pivot = df.pivot(
        index="batch_size",
        columns="num_threads",
        values="throughput_fps"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"{t} thread{'s' if t>1 else ''}"
                        for t in pivot.columns])
    ax.set_yticklabels([f"Batch {b}" for b in pivot.index])

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0f}\nFPS",
                    ha="center", va="center",
                    fontweight="bold", fontsize=11,
                    color="black" if val < pivot.values.max() * 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Throughput (FPS)")
    ax.set_title("Throughput Heatmap : Batch Size vs Thread Count",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Thread Count", fontsize=11)
    ax.set_ylabel("Batch Size", fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, "throughput_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")


# CHART 2: Latency Bar Chart (Mean / P95 / P99)

def plot_latency_bars(df, output_dir):
    df1 = df[df["batch_size"] == 1].sort_values("num_threads")

    labels = [f"{t} Thread{'s' if t>1 else ''}"
              for t in df1["num_threads"]]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - width, df1["latency_mean_ms"],
                   width, label="Mean",  color=COLORS[0], alpha=0.9)
    bars2 = ax.bar(x,         df1["latency_p95_ms"],
                   width, label="P95",   color=COLORS[1], alpha=0.9)
    bars3 = ax.bar(x + width, df1["latency_p99_ms"],
                   width, label="P99",   color=COLORS[2], alpha=0.9)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Inference Latency : Mean / P95 / P99 (Batch Size = 1)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.set_ylim(0, df1["latency_p99_ms"].max() * 1.3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "latency_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")

# CHART 3: Throughput Scaling Line Chart

def plot_throughput_scaling(df, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, batch in enumerate(sorted(df["batch_size"].unique())):
        subset = df[df["batch_size"] == batch].sort_values("num_threads")
        ax.plot(subset["num_threads"], subset["throughput_fps"],
                marker="o", linewidth=2.5, markersize=8,
                color=COLORS[i], label=f"Batch Size {batch}")

        for _, row in subset.iterrows():
            ax.annotate(f"{row['throughput_fps']:.0f}",
                        (row["num_threads"], row["throughput_fps"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, color=COLORS[i])

    ax.set_xlabel("Thread Count", fontsize=11)
    ax.set_ylabel("Throughput (FPS)", fontsize=11)
    ax.set_title("Throughput Scaling vs Thread Count",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(sorted(df["num_threads"].unique()))

    plt.tight_layout()
    path = os.path.join(output_dir, "throughput_scaling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")


# CHART 4: Latency vs Throughput Scatter

def plot_latency_vs_throughput(df, output_dir):
    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        df["latency_mean_ms"], df["throughput_fps"],
        c=df["num_threads"], cmap="viridis",
        s=120, zorder=5, edgecolors="white", linewidth=0.8
    )

    for _, row in df.iterrows():
        label = f"B{int(row['batch_size'])}T{int(row['num_threads'])}"
        ax.annotate(label,
                    (row["latency_mean_ms"], row["throughput_fps"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color="#374151")

    plt.colorbar(scatter, ax=ax, label="Thread Count")
    ax.set_xlabel("Mean Latency (ms)  Lower is better", fontsize=11)
    ax.set_ylabel("Throughput (FPS)   Higher is better", fontsize=11)
    ax.set_title("Latency vs Throughput : All Configurations\n"
                 "Label format: B=Batch Size, T=Threads",
                 fontsize=13, fontweight="bold", pad=15)
    ax.grid(alpha=0.3)

    best_thr = df.loc[df["throughput_fps"].idxmax()]
    best_lat = df.loc[df["latency_mean_ms"].idxmin()]

    ax.scatter(best_thr["latency_mean_ms"], best_thr["throughput_fps"],
               s=250, marker="*", color="#DC2626", zorder=6,
               label=f"Best throughput: {best_thr['throughput_fps']:.0f} FPS")
    ax.scatter(best_lat["latency_mean_ms"], best_lat["throughput_fps"],
               s=250, marker="*", color="#2563EB", zorder=6,
               label=f"Best latency: {best_lat['latency_mean_ms']:.1f} ms")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "latency_vs_throughput.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")


def main():
    print("=" * 50)
    print("  ML Benchmark Visualization")
    print("=" * 50)

    df = load_results(RESULTS_PATH)

    print("\n[INFO] Generating charts...")
    plot_throughput_heatmap(df, OUTPUT_DIR)
    plot_latency_bars(df, OUTPUT_DIR)
    plot_throughput_scaling(df, OUTPUT_DIR)
    plot_latency_vs_throughput(df, OUTPUT_DIR)

    print("\n[SUCCESS] All charts saved to results/")
    print("[INFO] Charts generated:")
    print("  - throughput_heatmap.png")
    print("  - latency_bars.png")
    print("  - throughput_scaling.png")
    print("  - latency_vs_throughput.png")

if __name__ == "__main__":
    main()