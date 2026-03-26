# ML Inference Benchmarking Tool

A cross-platform benchmarking tool for ML model inference using **ONNX Runtime**, built with **C++17**, **CMake**, and **Python**.

## What it does
- Loads a real ONNX model (MobileNetV2) via ONNX Runtime C++ API
- Runs inference across configurable batch sizes, thread counts, and precisions
- Measures latency (ms), throughput (inferences/sec), and memory usage (MB)
- Generates a benchmark report with charts

## Tech Stack
| Layer | Technology |
|---|---|
| Inference Engine | ONNX Runtime (C++ API) |
| Build System | CMake 3.20+ |
| Orchestration | Python 3.10+ |
| Visualization | matplotlib, pandas |
| CI/CD | GitHub Actions |

## Platform
- Windows (MSVC)
- Linux (GCC)

## Build Instructions

### Prerequisites
- Visual Studio 2022 with C++ workload (Windows)
- CMake 3.20+
- Python 3.10+

### Windows
```powershell
cmake -B build -S .
cmake --build build --config Release
```

### Run Benchmark
```powershell
python python/download_model.py
python python/run_benchmark.py
```

## Results
Benchmark reports are saved to the `results/` folder as JSON and PNG charts.