// ==============================================================
// Defines the BenchmarkRunner class interface.
//
//
//   BenchmarkResult: a plain data struct holding one run's
//     measurements (latency, throughput, memory)
//   BenchmarkRunner: loads an ONNX model and runs it
//     repeatedly to collect performance statistics
// ==============================================================

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include "onnxruntime_cxx_api.h"


struct BenchmarkConfig {
    int batch_size = 1;      // How many images to process at once
    int num_threads = 1;      // CPU threads for inference
    int iterations = 50;     // How many times to run inference
    int warmup_runs = 5;      // Runs before measuring (let CPU warm up)
};

// ==============================================================
// BenchmarkResult — measurements from one benchmark run
//
//
//   latency_ms    : How long ONE inference takes (user experience)
//   throughput    : How many inferences per second (server capacity)
//   memory_mb     : RAM used (critical for edge/mobile deployment)
//   These 3 together are the standard ML inference report card.
// ==============================================================
struct BenchmarkResult {
    BenchmarkConfig config;

    double latency_mean_ms = 0.0;
    double latency_min_ms = 0.0;
    double latency_max_ms = 0.0;
    double latency_p50_ms = 0.0;   // 50th percentile (median)
    double latency_p95_ms = 0.0;   // 95th percentile
    double latency_p99_ms = 0.0;   // 99th percentile (worst case)

    // Throughput
    double throughput_fps = 0.0;   // Inferences per second

    // Memory
    double memory_mb = 0.0;   // Peak memory usage in MB
};

// ==============================================================
// BenchmarkRunner 
// ==============================================================
class BenchmarkRunner {
public:
    explicit BenchmarkRunner(const std::string& model_path);

    BenchmarkResult run(const BenchmarkConfig& config);

    static void print_result(const BenchmarkResult& result);

private:
    Ort::Env            env_;
    Ort::SessionOptions session_options_;
    Ort::Session        session_;

    // Model metadata  : we read these from the loaded model
    std::string              input_name_;
    std::string              output_name_;
    std::vector<int64_t>     input_shape_;

    void                 configure_session(int num_threads);
    std::vector<float>   create_dummy_input(int batch_size);
    double               get_memory_usage_mb();

    // Stored model path for re-creating sessions
    std::string model_path_;
};