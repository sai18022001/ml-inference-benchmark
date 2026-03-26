// ==============================================================
// Implements the BenchmarkRunner class.
//
// 1. Constructor loads model then creates Ort::Session
// 2. run() creates dummy input then runs inference N times
// 3. Collects timing per iteration then computes statistics
// ==============================================================

#include "benchmark_runner.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

// ==============================================================
// Loads the ONNX model and reads its input/output metadata.
// ==============================================================
BenchmarkRunner::BenchmarkRunner(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "benchmark")
    , session_options_()
    , session_(nullptr)
    , model_path_(model_path)
{
    std::cout << "[INFO] Loading model: " << model_path << std::endl;

    // Basic session configuration
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );
#ifdef _WIN32
    // For Windows
    // Convert path to wide string
    std::wstring wide_path(model_path.begin(), model_path.end());

    // Load the model — this is where ONNX Runtime reads the file
    // and builds the computation graph in memory
    session_ = Ort::Session(env_, wide_path.c_str(), session_options_);
#else
    // For Linux Build
    session_ = Ort::Session(env_, model_path.c_str(), session_options_);
#endif

    // ==================================================
    // READ MODEL METADATA
    // ONNX models store input/output names and shapes
    // ==================================================
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input name
    auto input_name_ptr = session_.GetInputNameAllocated(0, allocator);
    input_name_ = std::string(input_name_ptr.get());

    // Get output name
    auto output_name_ptr = session_.GetOutputNameAllocated(0, allocator);
    output_name_ = std::string(output_name_ptr.get());

    // Get input shape
    // MobileNetV2: [1, 3, 224, 224] = batch, channels, height, width
    auto input_info = session_.GetInputTypeInfo(0);
    auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
    input_shape_ = tensor_info.GetShape();

    std::cout << "[INFO] Model loaded successfully!" << std::endl;
    std::cout << "[INFO] Input  name: " << input_name_ << std::endl;
    std::cout << "[INFO] Output name: " << output_name_ << std::endl;
    std::cout << "[INFO] Input shape: [";
    for (size_t i = 0; i < input_shape_.size(); i++) {
        std::cout << input_shape_[i];
        if (i < input_shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// ==============================================================
// CONFIGURE SESSION
// Set number of threads for this run..
// ==============================================================
void BenchmarkRunner::configure_session(int num_threads) {
    session_options_.SetIntraOpNumThreads(num_threads);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );
#ifdef _WIN32
    std::wstring wide_path(model_path_.begin(), model_path_.end());
    session_ = Ort::Session(env_, wide_path.c_str(), session_options_);
#else
    session_ = Ort::Session(env_, model_path_.c_str(), session_options_);
#endif
}

// ==============================================================
// Generates a random float tensor of the right shape.
// ==============================================================
std::vector<float> BenchmarkRunner::create_dummy_input(int batch_size) {
    // Calculate total elements: batch * channels * height * width
    // input_shape_ = [1, 3, 224, 224] for MobileNetV2
    size_t total = static_cast<size_t>(batch_size);
    for (size_t i = 1; i < input_shape_.size(); i++) {
        total *= static_cast<size_t>(input_shape_[i]);
    }
    // Fill with small random-ish values (normalized image range)
    std::vector<float> data(total, 0.5f);
    return data;
}

// ==============================================================
// GET MEMORY USAGE
// Returns current process memory usage in megabytes.
// Uses Windows API (PSAPI) 
// ==============================================================
double BenchmarkRunner::get_memory_usage_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

// ==============================================================
// the main benchmarking function
//
//   1. Configure session with requested thread count
//   2. Warmup runs (not measured) : stabilize CPU/cache
//   3. Measured runs : record each iteration's latency
//   4. Compute statistics from collected timings
// ==============================================================
BenchmarkResult BenchmarkRunner::run(const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.config = config;

    std::cout << "\n[BENCH] Starting benchmark..." << std::endl;
    std::cout << "[BENCH] Batch size:  " << config.batch_size << std::endl;
    std::cout << "[BENCH] Threads:     " << config.num_threads << std::endl;
    std::cout << "[BENCH] Iterations:  " << config.iterations << std::endl;
    std::cout << "[BENCH] Warmup runs: " << config.warmup_runs << std::endl;

    // Reconfigure session for this thread count
    configure_session(config.num_threads);

    // Prepare input tensor shape with actual batch size
    std::vector<int64_t> actual_shape = input_shape_;
    actual_shape[0] = config.batch_size;

    // Create memory info 
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    // Prepare input names and output names as C strings
    const char* input_names[] = { input_name_.c_str() };
    const char* output_names[] = { output_name_.c_str() };

    std::cout << "[BENCH] Running warmup..." << std::endl;
    for (int i = 0; i < config.warmup_runs; i++) {
        auto input_data = create_dummy_input(config.batch_size);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(), input_data.size(),
            actual_shape.data(), actual_shape.size()
        );
        session_.Run(
            Ort::RunOptions{ nullptr },
            input_names, &input_tensor, 1,
            output_names, 1
        );
    }

    // Record latency of each individual run.
    std::cout << "[BENCH] Running measured iterations..." << std::endl;
    std::vector<double> latencies;
    latencies.reserve(config.iterations);

    double memory_before = get_memory_usage_mb();

    for (int i = 0; i < config.iterations; i++) {
        auto input_data = create_dummy_input(config.batch_size);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(), input_data.size(),
            actual_shape.data(), actual_shape.size()
        );

        auto t_start = std::chrono::high_resolution_clock::now();

        session_.Run(
            Ort::RunOptions{ nullptr },
            input_names, &input_tensor, 1,
            output_names, 1
        );

        auto t_end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(
            t_end - t_start
        ).count();
        latencies.push_back(ms);
    }

    double memory_after = get_memory_usage_mb();

    // ==================================================
    // COMPUTE STATISTICS
    // Sort latencies to compute percentiles.
    // ==================================================
    std::sort(latencies.begin(), latencies.end());

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    result.latency_mean_ms = sum / latencies.size();
    result.latency_min_ms = latencies.front();
    result.latency_max_ms = latencies.back();

    // Percentiles via index into sorted array
    auto percentile = [&](double p) {
        size_t idx = static_cast<size_t>(p / 100.0 * latencies.size());
        idx = std::min(idx, latencies.size() - 1);
        return latencies[idx];
        };

    result.latency_p50_ms = percentile(50);
    result.latency_p95_ms = percentile(95);
    result.latency_p99_ms = percentile(99);

    // Throughput = how many inferences per second
    // Using mean latency: 1000ms / mean_ms * batch_size
    result.throughput_fps = (1000.0 / result.latency_mean_ms) * config.batch_size;

    // Memory delta
    result.memory_mb = memory_after - memory_before;

    return result;
}

// Print Results
void BenchmarkRunner::print_result(const BenchmarkResult& result) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  BENCHMARK RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Config:" << std::endl;
    std::cout << "    Batch size : " << result.config.batch_size << std::endl;
    std::cout << "    Threads    : " << result.config.num_threads << std::endl;
    std::cout << "    Iterations : " << result.config.iterations << std::endl;
    std::cout << "\n  Latency (ms):" << std::endl;
    std::cout << "    Mean  : " << result.latency_mean_ms << std::endl;
    std::cout << "    Min   : " << result.latency_min_ms << std::endl;
    std::cout << "    Max   : " << result.latency_max_ms << std::endl;
    std::cout << "    P50   : " << result.latency_p50_ms << std::endl;
    std::cout << "    P95   : " << result.latency_p95_ms << std::endl;
    std::cout << "    P99   : " << result.latency_p99_ms << std::endl;
    std::cout << "\n  Throughput  : " << result.throughput_fps << " FPS" << std::endl;
    std::cout << "  Memory delta: " << result.memory_mb << " MB" << std::endl;
    std::cout << "========================================\n" << std::endl;
}