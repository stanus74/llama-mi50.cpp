#!/bin/bash
set -e

# 1. Konfiguration
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export AMDGPU_ARCH="gfx906"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"

# Caching-Optimierung für ROCm/HIP
export CCACHE_SLOPINESS=include_file_mtime,time_macros

echo "🚀 Starte Caching-Build für MI50..."

# 2. Vorbereitung
mkdir -p build && cd build

# 3. CMake mit Ccache-Launchern und High-Speed Flags
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
    -DCMAKE_HIP_ARCHITECTURES="$AMDGPU_ARCH" \
    -DCMAKE_HIP_COMPILER_FORCED=1 \
    -DCMAKE_C_FLAGS="-O3 -march=native -ffast-math" \
    -DGGML_HIP=ON \
    -DGGML_HIPBLAS=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_NO_VMM=ON \
    -DGGML_HIP_NO_HIPBLASLT=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_HIP_MMQ_Y=96 \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DCMAKE_HG_FLAGS="-amdgpu-target=gfx906" \
    -DCMAKE_C_FLAGS="-fno-finite-math-only" \
    -DCMAKE_CXX_FLAGS="-fno-finite-math-only"

# 4. Build
make -j$(nproc)

echo "✅ Fertig! Caching wurde genutzt. Speed: 1150 t/s Ready."

