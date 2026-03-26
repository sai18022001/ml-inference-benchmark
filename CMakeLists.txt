cmake_minimum_required(VERSION 3.20)
project(ml_benchmark VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


#
# WIN32  = Windows (MSVC or MinGW)
# UNIX   = Linux and macOS

set(ONNXRUNTIME_VERSION "1.24.4")

if(WIN32)
    set(ONNXRUNTIME_DIR
        "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
    set(ONNXRUNTIME_LIB
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    set(ONNXRUNTIME_DLL
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
elseif(UNIX)
    set(ONNXRUNTIME_DIR
        "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
    set(ONNXRUNTIME_LIB
        "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    set(ONNXRUNTIME_DLL
        "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()

set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_DIR}/include")


if(NOT EXISTS "${ONNXRUNTIME_INCLUDE}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "\nONNX Runtime SDK not found at:\n  ${ONNXRUNTIME_DIR}\n"
        "Please download onnxruntime-${CMAKE_SYSTEM_NAME}-x64-${ONNXRUNTIME_VERSION}\n"
        "from https://github.com/microsoft/onnxruntime/releases\n"
        "and extract it into the third_party/ folder.\n"
    )
endif()

message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_DIR}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")

# Executables
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

# Compiler Warnings
if(MSVC)
    target_compile_options(ml_benchmark PRIVATE /W4)
else()
    target_compile_options(ml_benchmark PRIVATE -Wall -Wextra)
endif()

# Output Location
set_target_properties(ml_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)


add_custom_command(TARGET ml_benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL}"
        "$<TARGET_FILE_DIR:ml_benchmark>/"
    COMMENT "Copying ONNX Runtime shared library to output directory"
)