cmake_minimum_required(VERSION 3.20)
project(ml_benchmark VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.24.4")
set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")


if(NOT EXISTS "${ONNXRUNTIME_INCLUDE}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR 
        "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}\n"
        "Please download onnxruntime-win-x64-1.24.4 from:\n"
        "https://github.com/microsoft/onnxruntime/releases"
    )
endif()


add_executable(ml_benchmark
    cpp/main.cpp
    cpp/benchmark_runner.cpp
)



target_include_directories(ml_benchmark PRIVATE
    cpp/
    ${ONNXRUNTIME_INCLUDE}
)


target_link_libraries(ml_benchmark PRIVATE
    ${ONNXRUNTIME_LIB}
)


if(MSVC)
    target_compile_options(ml_benchmark PRIVATE /W4)
else()
    target_compile_options(ml_benchmark PRIVATE -Wall -Wextra)
endif()


set_target_properties(ml_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)


add_custom_command(TARGET ml_benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL}"
        "$<TARGET_FILE_DIR:ml_benchmark>/onnxruntime.dll"
    COMMENT "Copying onnxruntime.dll to output directory"
)