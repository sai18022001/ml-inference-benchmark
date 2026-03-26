#include "benchmark_runner.h"
#include <iostream>

BenchmarkRunner::BenchmarkRunner(const std::string& model_path) {
    std::cout << "[BenchmarkRunner] Model path: " << model_path << std::endl;
}

void BenchmarkRunner::run(int batch_size, int num_threads, int iterations) {
    std::cout << "[BenchmarkRunner] Running benchmark..." << std::endl;
    std::cout << "  Batch size:  " << batch_size << std::endl;
    std::cout << "  Threads:     " << num_threads << std::endl;
    std::cout << "  Iterations:  " << iterations << std::endl;
}