cmake_minimum_required(VERSION 3.20)

project(ml_benchmark VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ml_benchmark
    cpp/main.cpp
    cpp/benchmark_runner.cpp
)

target_include_directories(ml_benchmark PRIVATE
    cpp/
)

if(MSVC)
    target_compile_options(ml_benchmark PRIVATE /W4)
else()
    target_compile_options(ml_benchmark PRIVATE -Wall -Wextra)
endif()

set_target_properties(ml_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)