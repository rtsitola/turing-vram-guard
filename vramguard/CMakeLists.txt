cmake_minimum_required(VERSION 3.18)

# ── VS 2026 preview + CUDA 12.4 workaround (must be set before project()) ──
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(vramguard LANGUAGES CXX CUDA)

# ── CUDA architecture: Turing sm_75 (Quadro RTX 6000) ──
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ── CUDA Toolkit (must come before any target using CUDA::) ──
find_package(CUDAToolkit REQUIRED)

# ── Sources ──
add_library(vramguard SHARED
    src/vramguard.cpp
    src/detection.cu
)

# Standalone VMM validation test
add_executable(vmm_validation
    vmm_validation.cu
)
target_link_libraries(vmm_validation PRIVATE CUDA::cudart CUDA::cuda_driver)
target_include_directories(vmm_validation PRIVATE include)

# Targeted VMM probe on known-bad region
add_executable(vmm_probe
    vmm_probe.cu
)
target_link_libraries(vmm_probe PRIVATE CUDA::cudart CUDA::cuda_driver)
target_include_directories(vmm_probe PRIVATE include)

# Minimal: test ONE specific chunk with EXACT vram_mapper_detailed detection
add_executable(vmm_test_chunk
    vmm_test_chunk.cu
)
target_link_libraries(vmm_test_chunk PRIVATE CUDA::cudart)
target_include_directories(vmm_test_chunk PRIVATE include)

# Diagnostic: test detect_to_flags on cudaMalloc
add_executable(diag_detect
    diag_detect.cu
)
target_link_libraries(diag_detect PRIVATE CUDA::cudart)
target_include_directories(diag_detect PRIVATE include)

target_include_directories(vramguard PRIVATE include)

# ── CUDA Runtime ──
target_link_libraries(vramguard PRIVATE CUDA::cudart)

# ── Export all symbols (needed for ctypes to find vg_install) ──
set_target_properties(vramguard PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
target_compile_definitions(vramguard PRIVATE VRAMGUARD_EXPORTS)

# ── MSVC flags: C++ only, let CMake handle CUDA flags ──
target_compile_options(vramguard PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/W3>
)
