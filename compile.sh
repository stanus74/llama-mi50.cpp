#!/bin/bash
cat << 'EOF'

   ██╗     ██╗      █████╗ ███╗   ███╗ █████╗    ██████╗██████╗ ██████╗
   ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗  ██╔════╝██╔══██╗██╔══██╗
   ██║     ██║     ███████║██╔████╔██║███████║  ██║     ██████╔╝██████╔╝
   ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║  ██║     ██╔═══╝ ██╔═══╝
   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║  ╚██████╗██║     ██║
   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═════╝╚═╝     ╚═╝
            ██████╗ ███████╗██╗  ██╗ █████╗  ██████╗  ██████╗
           ██╔════╝ ██╔════╝╚██╗██╔╝██╔══██╗██╔═████╗██╔════╝
           ██║  ███╗█████╗   ╚███╔╝ ╚██████║██║██╔██║███████╗
           ██║   ██║██╔══╝   ██╔██╗  ╚═══██║████╔╝██║██╔═══██║
           ╚██████╔╝██║     ██╔╝ ██╗ █████╔╝╚██████╔╝╚██████╔╝
            ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚════╝  ╚═════╝  ╚═════╝

EOF

set -e

# 1. Check location
[[ ! -f "CMakeLists.txt" ]] && echo "Error: Not in llama.cpp root directory" && exit 1

# 2. Setup ROCm Environment Variables
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_PATH=$ROCM_PATH
export HIP_PLATFORM=amd
export HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}

if command -v amdgpu-arch &> /dev/null; then
    AMDGPU_ARCH=$(amdgpu-arch | head -n 1)
    echo "✓ Detected AMD GPU Architecture: $AMDGPU_ARCH"
else
    echo "Warning: amdgpu-arch tool not found. Defaulting to 'gfx906'."
    AMDGPU_ARCH="gfx906"
fi

# 3. Setup ccache for faster rebuilds
export CCACHE_SLOPINESS=include_file_mtime,time_macros
if command -v ccache &> /dev/null; then
    echo "✓ Using ccache for faster rebuilds"
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_HIP_COMPILER_LAUNCHER=ccache
else
    echo "⚠ ccache not found. Install with: apt install ccache"
fi

rm -rf build && mkdir -p build && cd build

# Setup logging (set LOG_ENABLED=1 to enable file logging)
LOG_ENABLED=${LOG_ENABLED:-0}
LOG_FILE="../build_$(date +%Y%m%d_%H%M%S).log"
WARNINGS_LOG="${LOG_FILE%.log}_warnings.log"
FILTERED_WARNINGS_LOG="${LOG_FILE%.log}_warnings_filtered.log"
if [ "$LOG_ENABLED" -eq 1 ]; then
    echo "📝 Build logs will be saved to: $LOG_FILE" >&2
fi

# CMake configuration (log to file)
# Optional: set GGML_HIP_MMQ_Y in the environment to pass -DGGML_HIP_MMQ_Y=<value>
if [ "$LOG_ENABLED" -eq 1 ]; then
    echo "=== CMAKE CONFIGURATION ===" | tee -a "$LOG_FILE"
    CMAKE_LOG_CMD="tee -a $LOG_FILE"
else
    echo "=== CMAKE CONFIGURATION ==="
    CMAKE_LOG_CMD="cat"
fi
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=$CMAKE_C_COMPILER_LAUNCHER \
    -DCMAKE_CXX_COMPILER_LAUNCHER=$CMAKE_CXX_COMPILER_LAUNCHER \
    -DCMAKE_HIP_COMPILER_LAUNCHER=$CMAKE_HIP_COMPILER_LAUNCHER \
    -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
    -DCMAKE_HIP_ARCHITECTURES="$AMDGPU_ARCH" \
    -DCMAKE_HIP_COMPILER_FORCED=1 \
    -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -DNDEBUG -ffast-math -fno-finite-math-only -ffp-contract=fast" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
    -DCMAKE_HIP_FLAGS="-Wno-ignored-attributes -Wno-cuda-compat -Wno-unused-result" \
    -DGGML_HIP=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_NO_VMM=ON \
    -DGGML_HIP_EXPORT_METRICS=ON \
    ${GGML_HIP_MMQ_Y:+-DGGML_HIP_MMQ_Y=$GGML_HIP_MMQ_Y} \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_FORCE_CUBLAS=OFF \
    -DGGML_CUDA_NO_PEER_COPY=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_CURL=ON \
    -DLLAMA_STATIC=OFF 2>&1 | eval "$CMAKE_LOG_CMD"

# Make build (log to file)
if [ "$LOG_ENABLED" -eq 1 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "=== BUILDING ===" | tee -a "$LOG_FILE"
    BUILD_LOG_CMD="tee -a $LOG_FILE"
else
    echo ""
    echo "=== BUILDING ==="
    BUILD_LOG_CMD="cat"
fi
make -j$(nproc) 2>&1 | eval "$BUILD_LOG_CMD"

BUILD_STATUS=$?
if [ "$LOG_ENABLED" -eq 1 ]; then
    echo "" | tee -a "$LOG_FILE"
else
    echo ""
fi

# Extract warnings (raw + filtered) from the build log
if [ "$LOG_ENABLED" -eq 1 ]; then
    grep -i "warning" "$LOG_FILE" > "$WARNINGS_LOG" || true
    grep -i "warning" "$LOG_FILE" | \
        grep -viE "loop not unrolled|gpu-max-threads-per-block|ISO C restricts enumerator values|no previous prototype" \
        > "$FILTERED_WARNINGS_LOG" || true
fi

if [ $BUILD_STATUS -eq 0 ]; then
    if [ "$LOG_ENABLED" -eq 1 ]; then
        echo "✅ Build complete!" | tee -a "$LOG_FILE"
        echo "   Binaries: ./build/bin/{llama-cli,llama-server,llama-bench}" | tee -a "$LOG_FILE"
        echo "   GPU Arch: $AMDGPU_ARCH" | tee -a "$LOG_FILE"
        echo "   Log file: $LOG_FILE" | tee -a "$LOG_FILE"
        echo "   Warnings: $WARNINGS_LOG" | tee -a "$LOG_FILE"
        echo "   Warnings (filtered): $FILTERED_WARNINGS_LOG" | tee -a "$LOG_FILE"
    else
        echo "✅ Build complete!"
        echo "   Binaries: ./build/bin/{llama-cli,llama-server,llama-bench}"
        echo "   GPU Arch: $AMDGPU_ARCH"
    fi
else
    if [ "$LOG_ENABLED" -eq 1 ]; then
        echo "❌ Build FAILED! See $LOG_FILE for details" | tee -a "$LOG_FILE"
    else
        echo "❌ Build FAILED!"
    fi
    exit 1
fi

