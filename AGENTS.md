# AGENTS.md - AMD MI50/GFX906 Optimization Guide

## Project Overview
This repository contains AMD MI50 (gfx906) specific optimizations for llama.cpp, a high-performance inference engine for LLMs. The optimizations target the CDNA/GCN architecture and include specialized CUDA/HIP kernels.

## Build Configuration

### Required Environment Variables
```bash
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_PATH=$ROCM_PATH
export HIP_PLATFORM=amd
export HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
export HCC_AMDGPU_TARGET=gfx906
```

### CMake Build Command
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$ROCM_PATH"/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER="$ROCM_PATH"/llvm/bin/clang++ \
    -DCMAKE_HIP_ARCHITECTURES="gfx906" \
    -DCMAKE_HIP_COMPILER_FORCED=1 \
    -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -DNDEBUG -ffast-math -fno-finite-math-only -ffp-contract=fast" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
    -DCMAKE_HIP_FLAGS="-Wno-ignored-attributes -Wno-cuda-compat -Wno-unused-result" \
    -DGGML_HIP=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_NO_VMM=ON \
    -DGGML_HIP_EXPORT_METRICS=ON \
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
    -DLLAMA_STATIC=OFF

cmake --build build -j"$(nproc)"
```

## MI50/GFX906 Optimization Categories

### A. GFX906 Kernel Files (Core Optimizations)
**Location**: `ggml/src/ggml-cuda/gfx906/`

New files containing MI50-specific kernels:
- `gfx906-common.cuh` - DPP-based warp reductions using AMD-specific instructions
- `gfx906-config.h` - Kernel configuration constants
- `gfx906-fattn-q8.cuh/cu` - Q8 Flash Attention optimized kernels
- `gfx906-mmvq-q4_0.cuh`, `gfx906-mmvq-q4_1.cuh`, `gfx906-mmvq-q8_0.cuh` - Warp-cooperative MMVQ
- `gfx906-mmq.cuh` - Optimized MMQ with vectorized loads
- `gfx906-mmq-prefetch.cuh` - Software pipelining for MMQ
- `gfx906-vecdotq.cuh` - Vectorized dot product operations

### B. Kernel Dispatch Integration
**Files**: `ggml/src/ggml-cuda/mmvq.cu`, `mmq.cuh`

Integration points for gfx906 kernels:
- Conditional includes: `#if defined(GGML_HIP_GFX906)`
- Dispatch logic selecting gfx906 path for small matrices (ncols_x <= 1024)
- Half-warp dispatch optimizations for MoE models

### C. Flash Attention Optimizations
**Files**: `ggml/src/ggml-cuda/fattn-common.cuh`, `fattn.cu`

Changes:
- Split-K optimization: Disabled for prompt processing (Q->ne[1] > 1), enabled for token generation
- Q8 tile kernel selection when K/V types are GGML_TYPE_Q8_0
- Head size validation: Supports 64, 96, 128, 256, 576 (excluding 40, 80, 112)

### D. Quantization Pipeline
**Files**: `ggml/src/ggml-cuda/mmq.cuh`, `vecdotq.cuh`

Optimizations:
- Software pipelining for Q8_0 loads with cache optimization
- MXFP4 load pipeline with e8m0 conversion
- Vectorized load functions for Q4_0 and Q4_1
- LDS conflict avoidance strategies

### E. Wave64/MoE Correctness Fixes
**Files**: `ggml/src/ggml-cuda/mmq.cu`, `mmid.cu`

Fixes for AMD Wave64 GPUs:
- Sub-warp shuffle fixes for large expert counts
- Fallback to generic path when n_expert_used >= warp_size/2
- MoE model stability improvements

### F. GPU Utility Functions
**File**: `ggml/src/ggml-cuda/common.cuh`

Unified infrastructure:
- `ggml_cuda_shfl_xor_sync()` - Dispatches to DPP on AMD, shuffle on NVIDIA
- `warp_reduce_amd_f32()` - Fused DPP reductions
- `gfx906_warp_reduce_sum_f32()` / `gfx906_warp_reduce_max_f32()`
- DPP-based shuffle operations: `hip_dpp_xor1/2/4/8/16()`

### G. Build System Integration
**File**: `ggml/src/ggml-cuda/CMakeLists.txt`

CMake additions:
```cmake
file(GLOB SRCS "gfx906/*.cuh")
list(APPEND GGML_HEADERS_CUDA ${SRCS})
file(GLOB SRCS "gfx906/*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
file(GLOB SRCS "gfx906/fused/*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
file(GLOB SRCS "gfx906/attention/*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
```

### H. Utility Scripts
**Location**: Root directory

- `SCRIPT_compile_MI50.sh` - Compilation script with all flags
- `SCRIPT_llama_bench.sh` - Benchmarking script
- `SCRIPT_launch_server_MI50.sh` - Server launch configuration
- `SCRIPT_overclock_upp_MI50.sh` - GPU overclocking (use with caution)

## Key Technical Details

### DPP (Data Parallel Primitives) Optimizations
The gfx906-common.cuh file implements fused DPP operations:
- `DEFINE_FUSED_DPP_F32` macro for generating DPP shuffle+ALU instructions
- Reduces instruction count by fusing shuffle with add/max operations
- Uses inline assembly for maximum performance

### Configuration Constants (gfx906-config.h)
```c
#define GFX906_MMQ_ITER_K 256
#define GFX906_MMQ_NWARPS 2
#define GFX906_FATTN_Q8_ENABLED 1
#define GFX906_USE_DPP_REDUCTIONS 1
```

### Memory Optimization Strategies
1. **LDS Bank Conflict Avoidance**: Careful stride selection (tested stride 40 vs 41)
2. **Software Pipelining**: Async loads followed by compute
3. **Vectorized Memory Access**: Using int4/float4 for bandwidth efficiency
4. **Register Pressure Management**: Overlaying KQ_acc onto V_tmp shared memory

## Testing & Verification

### Runtime Verification
```bash
HIP_VISIBLE_DEVICES=0 GGML_HIP_DEBUG=1 ./build/bin/llama-bench -m model.gguf
```

Look for:
- gfx906 kernel path indicators in logs
- Tile configuration messages
- Q8 attention kernel selection

### Benchmark Parameters
```bash
./build/bin/llama-bench \
    -m model.gguf \
    -ngl 99 \
    -t $(nproc) \
    -fa 1 \
    -ctk q8_0 \
    -ctv f16 \
    --main-gpu 0 \
    -p 512,2048,8192 \
    -n 1,128,2048
```

## Portability Notes

When porting to newer llama.cpp versions (e.g., b7871+):
- Categories D, B, G have highest conflict probability (MMQ/CDNA refactors)
- Categories A, F, C are most portable
- Check for new MMF/CDNA backend that may replace some optimizations

## Important Compilation Notes

1. **ROCm Version**: Optimized for ROCm 6.x
2. **Compiler**: Must use ROCm clang/clang++
3. **Architecture**: Must specify gfx906 explicitly
4. **VMM**: Disable Virtual Memory Management for MI50 compatibility
5. **No hipBLASLt**: May need to disable if experiencing crashes

## Environment for Execution

```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export GGML_BACKEND_HIP=1
export HCC_AMDGPU_TARGET=gfx906
```

## Safety & Stability

- The optimizations include MoE stability fixes for large models
- Wave64 sub-warp shuffle fixes prevent incorrect behavior
- LDS write conflict fixes in need_check paths
- Test thoroughly with your specific model architecture

## Recommended Patch Application Order

```
F → DPP Warp Utils (foundation)
C → Flash Attention (core feature)
E → Wave64 Fix (stability)
G → Build-System (compilation)
A → gfx906 Kernels (implementation)
B → Dispatch (integration)
D → Q8 Pipeline (optimization)
H → Scripts (optional tooling)
```
