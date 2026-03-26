# ML Inference Benchmarking Tool

![Build Status](https://github.com/sai18022001/ml-inference-benchmark/actions/workflows/ci.yml/badge.svg)

A cross-platform benchmarking tool for ML model inference using **ONNX Runtime**,
built with **C++17**, **CMake**, and **Python**.

Measures latency, throughput, and memory usage of neural network inference
across different batch sizes and thread configurations — the kind of performance
analysis done on ML hardware at companies like Qualcomm, Microsoft, and Google.

## Results — MobileNetV2 on CPU

| Config | Mean Latency | P99 Latency | Throughput |
|--------|-------------|-------------|------------|
| Batch=1, Threads=1 | 7.47 ms | 9.58 ms | 133.8 FPS |
| Batch=1, Threads=2 | 5.00 ms | 6.90 ms | 199.8 FPS |
| Batch=1, Threads=4 | 3.25 ms | 3.85 ms | 307.2 FPS |
| Batch=4, Threads=4 | 10.30 ms | 11.93 ms | 388.3 FPS |
| **Batch=8, Threads=4** | **19.54 ms** | **22.43 ms** | **409.4 FPS** |

> Best throughput: **409 FPS** (Batch=8, Threads=4)
> Best latency: **3.25ms** (Batch=1, Threads=4)

## Architecture
```
ml-inference-benchmark/
??? cpp/                    # C++ inference engine
?   ??? main.cpp            # Entry point + config sweep
?   ??? benchmark_runner.h  # BenchmarkResult / BenchmarkConfig structs
?   ??? benchmark_runner.cpp# ONNX Runtime inference + timing
??? python/
?   ??? download_model.py   # Downloads MobileNetV2 ONNX model
?   ??? run_benchmark.py    # Full pipeline orchestration
?   ??? visualize.py        # Generates benchmark charts
??? CMakeLists.txt          # Cross-platform build (Windows + Linux)
??? .github/workflows/
    ??? ci.yml              # GitHub Actions CI
```

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Inference Engine | ONNX Runtime 1.24.4 (C++ API) |
| Build System | CMake 3.20+ |
| Compiler | MSVC (Windows) / GCC (Linux) |
| Orchestration | Python 3.12+ |
| Visualization | matplotlib, pandas |
| CI/CD | GitHub Actions (Windows + Linux) |

## Quick Start

### Prerequisites
- Visual Studio 2022 with C++ workload (Windows) or GCC (Linux)
- CMake 3.20+
- Python 3.10+
- ONNX Runtime SDK — download from
  [releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.4)
  and extract to `third_party/`

### Build (Windows)
```powershell
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Build (Linux)
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Run Full Pipeline
```powershell
py python/run_benchmark.py
```

This will download the model, run the benchmark sweep, and generate charts in `results/`.

## Benchmark Charts

The tool generates 4 charts automatically:

- **Throughput Heatmap** — batch size vs thread count grid
- **Latency Bars** — Mean / P95 / P99 per thread count
- **Throughput Scaling** — how FPS grows with more threads
- **Latency vs Throughput** — the classic ML inference tradeoff plot

## Key Engineering Decisions

**Why ONNX Runtime?** Industry standard inference engine used by Microsoft,
Qualcomm, and others. Supports CPU, GPU, and specialized accelerators via
execution providers.

**Why P95/P99 latency?** Mean latency hides tail latency. Production SLAs
are defined by worst-case behavior — P99 tells you what 99% of your users
actually experience.

**Why warmup runs?** First inference runs are slower due to cold CPU caches
and memory paging. Discarding warmup runs gives stable, reproducible measurements.