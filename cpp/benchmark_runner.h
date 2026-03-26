#pragma once

#include <string>
#include <vector>

class BenchmarkRunner {
public:
    BenchmarkRunner(const std::string& model_path);
    void run(int batch_size, int num_threads, int iterations);
};