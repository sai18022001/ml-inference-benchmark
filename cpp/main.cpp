// ==============================================================
// Runs a full benchmark sweep across multiple
// configurations and saves results to JSON.
// ==============================================================

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <ctime>
#include "onnxruntime_cxx_api.h"
#include "benchmark_runner.h"

// ==============================================================
// Save results to a JSON file
// ==============================================================
void save_results_json(
    const std::vector<BenchmarkResult>& results,
    const std::string& output_path)
{
    std::ofstream f(output_path);
    if (!f.is_open()) {
        std::cerr << "[ERROR] Cannot write to: " << output_path << std::endl;
        return;
    }

    f << "[\n";
    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        f << "  {\n";
        f << "    \"batch_size\": " << r.config.batch_size << ",\n";
        f << "    \"num_threads\": " << r.config.num_threads << ",\n";
        f << "    \"iterations\": " << r.config.iterations << ",\n";
        f << "    \"latency_mean_ms\": " << r.latency_mean_ms << ",\n";
        f << "    \"latency_min_ms\": " << r.latency_min_ms << ",\n";
        f << "    \"latency_max_ms\": " << r.latency_max_ms << ",\n";
        f << "    \"latency_p50_ms\": " << r.latency_p50_ms << ",\n";
        f << "    \"latency_p95_ms\": " << r.latency_p95_ms << ",\n";
        f << "    \"latency_p99_ms\": " << r.latency_p99_ms << ",\n";
        f << "    \"throughput_fps\": " << r.throughput_fps << ",\n";
        f << "    \"memory_mb\": " << r.memory_mb << "\n";
        f << "  }";
        if (i < results.size() - 1) f << ",";
        f << "\n";
    }
    f << "]\n";

    std::cout << "[INFO] Results saved to: " << output_path << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "   ML Inference Benchmarking Tool v1.0          " << std::endl;
    std::cout << "   ONNX Runtime "
        << OrtGetApiBase()->GetVersionString() << std::endl;
    std::cout << "================================================" << std::endl;

    std::string model_path = "models/mobilenetv2.onnx";
    std::string results_dir = "results";
    if (argc > 1) model_path = argv[1];
    if (argc > 2) results_dir = argv[2];

    try {
        BenchmarkRunner runner(model_path);

        // ==================================================
        // CONFIGURATION SWEEP
        //
        // batch_sizes: 1 = single inference (low latency mode)
        //              4, 8 = batched (high throughput mode)
        // threads: 1, 2, 4 = how many CPU cores to use
        // ==================================================
        std::vector<int> batch_sizes = { 1, 4, 8 };
        std::vector<int> thread_counts = { 1, 2, 4 };

        std::vector<BenchmarkResult> all_results;

        for (int batch : batch_sizes) {
            for (int threads : thread_counts) {
                BenchmarkConfig config;
                config.batch_size = batch;
                config.num_threads = threads;
                config.iterations = 50;
                config.warmup_runs = 5;

                BenchmarkResult result = runner.run(config);
                BenchmarkRunner::print_result(result);
                all_results.push_back(result);
            }
        }

        // ==================================================
        // SAVE RESULTS
        // ==================================================
        std::string output_path = results_dir + "/benchmark_results.json";
        save_results_json(all_results, output_path);

        std::cout << "\n[INFO] Full sweep complete!" << std::endl;
        std::cout << "[INFO] Total configurations tested: "
            << all_results.size() << std::endl;

    }
    catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] ONNX Runtime: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}