Alle Optimierungen für die MI50 GFX906 GPU an der Original llama.cpp Version B7676

diff --git a/ggml/src/ggml-cuda/CMakeLists.txt b/ggml/src/ggml-cuda/CMakeLists.txt
index d313c1ac9..dcc004134 100644
--- a/ggml/src/ggml-cuda/CMakeLists.txt
+++ b/ggml/src/ggml-cuda/CMakeLists.txt
@@ -47,10 +47,7 @@ if (CUDAToolkit_FOUND)
                 #     check Modules/Internal/CMakeCUDAArchitecturesValidate.cmake in the CMake git repository instead.
                 # However, the architectures 120a-real and 121a-real should work with basically any CMake version and
                 #     until the release of e.g. Rubin there is no benefit to shipping virtual architectures for Blackwell.
-                list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real)
-            endif()
-            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
-                list(APPEND CMAKE_CUDA_ARCHITECTURES 121a-real)
+                list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real 121a-real)
             endif()
         endif()
     endif()
diff --git a/ggml/src/ggml-cuda/add-id.cu b/ggml/src/ggml-cuda/add-id.cu
index 8d9cf692b..16047f488 100644
--- a/ggml/src/ggml-cuda/add-id.cu
+++ b/ggml/src/ggml-cuda/add-id.cu
@@ -1,13 +1,12 @@
 #include "add-id.cuh"
 
-static __global__ void add_id_kernel(
+static __global__ void add_id_kernel_reference(
         const float * src0, const float * src1, const int32_t * src2, float * dst,
         int64_t ne0, int64_t ne1,
         size_t nb01, size_t nb02,
         size_t nb11,
         size_t nb21
     ) {
-
     const int64_t i1 = blockIdx.x;
     const int64_t i2 = blockIdx.y;
 
@@ -25,6 +24,66 @@ static __global__ void add_id_kernel(
     }
 }
 
+static __global__ void add_id_kernel_vec4(
+        const float * __restrict__ src0,
+        const float * __restrict__ src1,
+        const int32_t * __restrict__ src2,
+        float * __restrict__ dst,
+        const int ne0,
+        const int ne01,
+        const int s0_stride,
+        const int s0_stride2,
+        const int s1_stride,
+        const int s2_stride
+    ) {
+    const int i1 = blockIdx.x;
+    const int i2 = blockIdx.y;
+
+    const int i11 = src2[i1 + i2 * s2_stride];
+
+    const int src0_offset = i1 * s0_stride + i2 * s0_stride2;
+    const int src1_offset = i11 * s1_stride;
+    const int dst_offset = i1 * ne0 + i2 * ne01 * ne0;
+
+    const float4 * __restrict__ src0_vec = reinterpret_cast<const float4 *>(src0 + src0_offset);
+    const float4 * __restrict__ src1_vec = reinterpret_cast<const float4 *>(src1 + src1_offset);
+    float4 * __restrict__ dst_vec = reinterpret_cast<float4 *>(dst + dst_offset);
+
+    const int ne0_vec = ne0 >> 2;
+
+    for (int i0 = threadIdx.x; i0 < ne0_vec; i0 += blockDim.x) {
+        const float4 a = src0_vec[i0];
+        const float4 b = src1_vec[i0];
+        dst_vec[i0] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
+    }
+}
+
+static __global__ void add_id_kernel_contiguous(
+        const float * __restrict__ src0,
+        const float * __restrict__ src1,
+        const int32_t * __restrict__ src2,
+        float * __restrict__ dst,
+        const int ne0,
+        const int ne01,
+        const int s0_stride,
+        const int s0_stride2,
+        const int s1_stride,
+        const int s2_stride
+    ) {
+    const int i1 = blockIdx.x;
+    const int i2 = blockIdx.y;
+
+    const int i11 = src2[i1 + i2 * s2_stride];
+
+    const float * __restrict__ src0_row = src0 + i1 * s0_stride + i2 * s0_stride2;
+    const float * __restrict__ src1_row = src1 + i11 * s1_stride;
+    float * __restrict__ dst_row = dst + i1 * ne0 + i2 * ne01 * ne0;
+
+    for (int i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
+        dst_row[i0] = src0_row[i0] + src1_row[i0];
+    }
+}
+
 void ggml_cuda_op_add_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     const ggml_tensor * src0 = dst->src[0];
     const ggml_tensor * src1 = dst->src[1];
@@ -46,13 +105,49 @@ void ggml_cuda_op_add_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     const int32_t * src2_d = (const int32_t *)src2->data;
     float * dst_d = (float *)dst->data;
 
-    int threads = std::min((int)ne00, 768); // cols
-    dim3 blocks(ne01, ne02); // n_experts_used, n_tokens
-    add_id_kernel<<<blocks, threads, 0, ctx.stream()>>>(
-        src0_d, src1_d, src2_d, dst_d,
-        ne0, ne1,
-        nb01, nb02,
-        nb11,
-        nb21
-    );
+    cudaStream_t stream = ctx.stream();
+
+    const bool is_contiguous = (nb01 == ne00 * sizeof(float)) &&
+                               (nb11 == ne10 * sizeof(float));
+
+    const bool is_aligned = ((uintptr_t)src0_d % 16 == 0) &&
+                            ((uintptr_t)src1_d % 16 == 0) &&
+                            ((uintptr_t)dst_d  % 16 == 0);
+
+    const bool can_vectorize = is_contiguous && is_aligned && (ne00 % 4 == 0) && (ne00 <= INT_MAX);
+
+    const dim3 blocks(ne01, ne02);
+
+    if (can_vectorize) {
+        const int threads_vec4 = std::min((int)(ne00 / 4), 768);
+        add_id_kernel_vec4<<<blocks, threads_vec4, 0, stream>>>(
+            src0_d, src1_d, src2_d, dst_d,
+            (int)ne00,
+            (int)ne01,
+            (int)(nb01 / sizeof(float)),
+            (int)(nb02 / sizeof(float)),
+            (int)(nb11 / sizeof(float)),
+            (int)(nb21 / sizeof(int32_t))
+        );
+    } else if (is_contiguous && ne00 <= INT_MAX) {
+        const int threads = std::min((int)ne00, 768);
+        add_id_kernel_contiguous<<<blocks, threads, 0, stream>>>(
+            src0_d, src1_d, src2_d, dst_d,
+            (int)ne00,
+            (int)ne01,
+            (int)(nb01 / sizeof(float)),
+            (int)(nb02 / sizeof(float)),
+            (int)(nb11 / sizeof(float)),
+            (int)(nb21 / sizeof(int32_t))
+        );
+    } else {
+        const int threads = std::min((int)ne00, 768);
+        add_id_kernel_reference<<<blocks, threads, 0, stream>>>(
+            src0_d, src1_d, src2_d, dst_d,
+            ne0, ne1,
+            nb01, nb02,
+            nb11,
+            nb21
+        );
+    }
 }
diff --git a/ggml/src/ggml-cuda/common.cuh b/ggml/src/ggml-cuda/common.cuh
index 9516d8ec8..b547e27d9 100644
--- a/ggml/src/ggml-cuda/common.cuh
+++ b/ggml/src/ggml-cuda/common.cuh
@@ -262,6 +262,10 @@ static const char * cu_get_error_str(CUresult err) {
 #define FLASH_ATTN_AVAILABLE
 #endif // !defined(GGML_CUDA_NO_FA) && !(defined(GGML_USE_MUSA) && __MUSA_ARCH__ < 220)
 
+#if defined(TURING_MMA_AVAILABLE)
+#define LDMATRIX_TRANS_AVAILABLE
+#endif // defined(TURING_MMA_AVAILABLE)
+
 static bool fp16_available(const int cc) {
     return ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_PASCAL ||
         (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_PH1);
@@ -397,6 +401,35 @@ struct ggml_cuda_unroll<1> {
     }
 };
 
+// ================================================================================================
+// AMD GFX906 DPP-based Warp Reductions
+// ================================================================================================
+// All AMD GFX906-specific DPP optimizations moved to gfx906/gfx906-common.cuh
+// ================================================================================================
+
+#ifdef GGML_USE_HIP
+    #include "gfx906/gfx906-common.cuh"
+#endif // GGML_USE_HIP
+
+// ============================================================================
+// Unified shuffle XOR operation - dispatches to DPP on AMD, shuffle on NVIDIA
+// ============================================================================
+template<int width = WARP_SIZE, typename T>
+static __device__ __forceinline__ T ggml_cuda_shfl_xor_sync(T x, int offset) {
+#if defined(GGML_USE_HIP)
+    switch (~offset) {
+        case ~1:  return hip_dpp_xor1(x);
+        case ~2:  return hip_dpp_xor2(x);
+        case ~4:  return hip_dpp_xor4(x);
+        case ~8:  return hip_dpp_xor8(x);
+        case ~16: return hip_dpp_xor16(x);
+        default:  return __shfl_xor(x, offset, width);
+    }
+#else
+    return __shfl_xor_sync(0xffffffff, x, offset, width);
+#endif
+}
+
 template<int width = WARP_SIZE>
 static __device__ __forceinline__ int warp_reduce_sum(int x) {
 #if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
@@ -404,7 +437,7 @@ static __device__ __forceinline__ int warp_reduce_sum(int x) {
 #else
 #pragma unroll
     for (int offset = width/2; offset > 0; offset >>= 1) {
-        x += __shfl_xor_sync(0xffffffff, x, offset, width);
+        x += ggml_cuda_shfl_xor_sync<width>(x, offset);
     }
     return x;
 #endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
@@ -412,19 +445,23 @@ static __device__ __forceinline__ int warp_reduce_sum(int x) {
 
 template<int width = WARP_SIZE>
 static __device__ __forceinline__ float warp_reduce_sum(float x) {
+#if defined(GGML_USE_HIP)
+    return warp_reduce_amd_f32<width, AddOp>(x);  // Use fused DPP instructions
+#else
 #pragma unroll
     for (int offset = width/2; offset > 0; offset >>= 1) {
-        x += __shfl_xor_sync(0xffffffff, x, offset, width);
+        x += ggml_cuda_shfl_xor_sync<width>(x, offset);
     }
     return x;
+#endif
 }
 
 template<int width = WARP_SIZE>
 static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
 #pragma unroll
     for (int offset = width/2; offset > 0; offset >>= 1) {
-        a.x += __shfl_xor_sync(0xffffffff, a.x, offset, width);
-        a.y += __shfl_xor_sync(0xffffffff, a.y, offset, width);
+        a.x += ggml_cuda_shfl_xor_sync<width>(a.x, offset);
+        a.y += ggml_cuda_shfl_xor_sync<width>(a.y, offset);
     }
     return a;
 }
@@ -434,10 +471,9 @@ static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
 #ifdef FP16_AVAILABLE
 #pragma unroll
     for (int offset = width/2; offset > 0; offset >>= 1) {
-        a = __hadd2(a, __shfl_xor_sync(0xffffffff, a, offset, width));
+        a = __hadd2(a, ggml_cuda_shfl_xor_sync<width>(a, offset));
     }
     return a;
-
 #else
     NO_DEVICE_CODE;
     return a;
@@ -451,7 +487,7 @@ static __device__ __forceinline__ int warp_reduce_all(int x) {
     } else {
 #pragma unroll
         for (int offset = width/2; offset > 0; offset >>= 1) {
-            x = __shfl_xor_sync(0xffffffff, x, offset, width) && x;
+            x = ggml_cuda_shfl_xor_sync<width>(x, offset) && x;
         }
         return x;
     }
@@ -464,7 +500,7 @@ static __device__ __forceinline__ int warp_reduce_any(int x) {
     } else {
 #pragma unroll
         for (int offset = width/2; offset > 0; offset >>= 1) {
-            x = __shfl_xor_sync(0xffffffff, x, offset, width) || x;
+            x = ggml_cuda_shfl_xor_sync<width>(x, offset) || x;
         }
         return x;
     }
@@ -472,11 +508,15 @@ static __device__ __forceinline__ int warp_reduce_any(int x) {
 
 template<int width = WARP_SIZE>
 static __device__ __forceinline__ float warp_reduce_max(float x) {
+#if defined(GGML_USE_HIP)
+    return warp_reduce_amd_f32<width, MaxOp>(x);  // Use fused DPP instructions
+#else
 #pragma unroll
     for (int offset = width/2; offset > 0; offset >>= 1) {
-        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
+        x = fmaxf(x, ggml_cuda_shfl_xor_sync<width>(x, offset));
     }
     return x;
+#endif
 }
 
 template<typename T, int width = WARP_SIZE>
@@ -560,7 +600,7 @@ static __device__ __forceinline__ half2 warp_reduce_max(half2 x) {
 #if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL || defined(GGML_USE_HIP)
 #pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
-       x = ggml_cuda_hmax2(x, __shfl_xor_sync(0xffffffff, x, offset, width));
+       x = ggml_cuda_hmax2(x, ggml_cuda_shfl_xor_sync<width>(x, offset));
    }
    return x;
 #else
@@ -700,6 +740,10 @@ static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
 #if CUDART_VERSION >= 12080
     const nv_bfloat16 e = __nv_cvt_e8m0_to_bf16raw(x);
     return (float) e;
+#elif defined(GGML_USE_HIP) && defined(__gfx906__)
+    // GFX906: Branchless with direct bit cast
+    const uint32_t bits = x ? ((uint32_t)x << 23) : 0x00400000u;
+    return __uint_as_float(bits);
 #else
     uint32_t bits;
     if (x == 0) {
@@ -1036,7 +1080,7 @@ struct ggml_tensor_extra_gpu {
 #define USE_CUDA_GRAPH
 #endif
 
-struct ggml_cuda_graph_node_properties {
+struct ggml_graph_node_properties {
     void * node_address;
     ggml_op node_op;
     int64_t ne[GGML_MAX_DIMS];
@@ -1061,25 +1105,10 @@ struct ggml_cuda_graph {
     std::vector<cudaGraphNode_t> nodes;
     bool disable_due_to_gpu_arch = false;
     bool disable_due_to_too_many_updates = false;
+    bool disable_due_to_failed_graph_capture = false;
     int number_consecutive_updates = 0;
-    std::vector<ggml_cuda_graph_node_properties> props;
-
-    void record_update(bool use_graph, bool update_required) {
-        if (use_graph && update_required) {
-            number_consecutive_updates++;
-        } else {
-            number_consecutive_updates = 0;
-        }
-        if (number_consecutive_updates >= 4) {
-            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
-            disable_due_to_too_many_updates = true;
-        }
-    }
-
-    bool is_enabled() const {
-        static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);
-        return !(disable_due_to_gpu_arch || disable_cuda_graphs_due_to_env || disable_due_to_too_many_updates);
-    }
+    bool cuda_graphs_enabled = false;
+    std::vector<ggml_graph_node_properties> ggml_graph_properties;
 #endif
 };
 
diff --git a/ggml/src/ggml-cuda/fattn-common.cuh b/ggml/src/ggml-cuda/fattn-common.cuh
index 314467872..73872a4ca 100644
--- a/ggml/src/ggml-cuda/fattn-common.cuh
+++ b/ggml/src/ggml-cuda/fattn-common.cuh
@@ -11,12 +11,11 @@
 #define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
 
 // log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
-//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
+//     by the VKQ accumulators is effectively being shifted up by a factor of 8.
 // This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
 // However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
-// Still, the value range should be shifted as much as necessary but as little as possible.
-// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
-#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)
+#define FATTN_KQ_MAX_OFFSET 0.6931f
+
 
 typedef void (* fattn_kernel_t)(
         const char * __restrict__ Q,
@@ -276,8 +275,8 @@ static __device__ __forceinline__ void quantize_q8_1_to_shared(
     }
 #pragma unroll
     for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
-        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
-        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
+        amax = fmaxf(amax, ggml_cuda_shfl_xor_sync<32>(amax, mask));
+        sum +=             ggml_cuda_shfl_xor_sync<32>(sum,  mask);
     }
 
     const float d = amax / 127;
@@ -914,7 +913,7 @@ void launch_fattn(
 
         const int nblocks_stream_k = max_blocks;
 
-        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || tiles_efficiency_percent < 75;
+        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75;
 
         blocks_num.x = use_stream_k ? nblocks_stream_k : ntiles_total;
         blocks_num.y = 1;
@@ -951,6 +950,20 @@ void launch_fattn(
             }
         }
 
+        // AMD GFX906 optimization: Different Split-K for PP vs TG
+        const bool is_amd = !GGML_CUDA_CC_IS_NVIDIA(cc);
+        const bool is_prompt_processing = Q->ne[1] > 1;  // Q->ne[1] = num query tokens
+
+        if (is_amd) {
+            if (is_prompt_processing) {
+                // PP: Disable Split-K to avoid combine overhead
+                parallel_blocks = 1;
+            } else {
+                // TG: Use auto-tuned value for better SM utilization
+                // (parallel_blocks already set by auto-tuner)
+            }
+        }
+
         blocks_num.x = ntiles_x;
         blocks_num.y = parallel_blocks;
         blocks_num.z = (Q->ne[2]/ncols2)*Q->ne[3];
diff --git a/ggml/src/ggml-cuda/fattn-mma-f16.cuh b/ggml/src/ggml-cuda/fattn-mma-f16.cuh
index 856291dc3..e53bbc050 100644
--- a/ggml/src/ggml-cuda/fattn-mma-f16.cuh
+++ b/ggml/src/ggml-cuda/fattn-mma-f16.cuh
@@ -98,6 +98,19 @@ static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_co
     return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
 }
 
+static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_rdna(const int DKQ, const int DV, const int ncols) {
+    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2,  64, 128, 128, 128, 2, true);
+    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 2, true);
+    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 2, true);
+
+    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
+    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
+    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);
+
+    // TODO tune specifically for RDNA
+    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
+}
+
 static __host__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
     if (ampere_mma_available(cc)) {
         return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
@@ -105,6 +118,9 @@ static __host__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, c
     if (turing_mma_available(cc)) {
         return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
     }
+    if (amd_wmma_available(cc)) {
+        return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
+    }
     GGML_ASSERT(volta_mma_available(cc));
     return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
 }
@@ -116,6 +132,8 @@ static constexpr __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config(cons
     return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
 #elif defined(VOLTA_MMA_AVAILABLE)
     return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
+#elif defined(AMD_WMMA_AVAILABLE)
+    return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
 #else
     GGML_UNUSED_VARS(DKQ, DV, ncols);
     return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
@@ -186,6 +204,23 @@ static constexpr __device__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ,
     return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).Q_in_reg;
 }
 
+static constexpr __device__ int get_cols_per_thread() {
+#if defined(AMD_WMMA_AVAILABLE)
+    return 1; // RDNA has a single column.
+#else
+    return 2; // This is specifically KQ columns, Volta only has a single VKQ column.
+#endif // defined(AMD_WMMA_AVAILABLE)
+}
+
+static __host__ int get_cols_per_warp(const int cc) {
+    if (turing_mma_available(cc) || amd_wmma_available(cc)) {
+        return 16;
+    } else {
+        // Volta
+        return 32;
+    }
+}
+
 // ------------------------------------------------------------------------------------------------------------------
 
 static __host__ int ggml_cuda_fattn_mma_get_nstages(const int DKQ, const int DV, const int ncols1, const int ncols2, const int cc) {
@@ -393,10 +428,10 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
         const int jt,
         const int kb0,
         const int k_VKQ_sup) {
-#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
+#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4))
     constexpr int  ncols           = ncols1 * ncols2;
     constexpr int  cols_per_warp   = T_B_KQ::I;
-    constexpr int  cols_per_thread = 2; // This is specifically KQ columns, Volta only has a single VKQ column.
+    constexpr int  cols_per_thread = get_cols_per_thread();
     constexpr int  np              = nwarps * (cols_per_warp/ncols2) / ncols1; // Number of parallel CUDA warps per Q column.
     constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
     constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols);
@@ -413,6 +448,8 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
     const int k_VKQ_0 = kb0 * nbatch_fa;
 #if defined(TURING_MMA_AVAILABLE)
     T_C_KQ KQ_C[nbatch_fa/(np*(cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))];
+#elif defined(AMD_WMMA_AVAILABLE)
+    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
 #else // Volta
     T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
 #endif // defined(TURING_MMA_AVAILABLE)
@@ -461,8 +498,14 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
                     if constexpr (cols_per_warp == 8) {
                         mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
                     } else {
-                        // Wide version of KQ_C is column-major => swap A and B.
+                        // Wide version of KQ_C is column-major
+#if defined(AMD_WMMA_AVAILABLE)
+                        // RDNA matrix C is column-major.
+                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
+#else
+                        // swap A and B for CUDA.
                         mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[k_KQ_0/T_A_KQ::J], K_A);
+#endif // defined(AMD_WMMA_AVAILABLE)
                     }
                 }
             }
@@ -479,8 +522,14 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
                     T_A_KQ K_A;
                     load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);
 
-                    // Wide version of KQ_C is column-major => swap A and B.
+                    // Wide version of KQ_C is column-major
+#if defined(AMD_WMMA_AVAILABLE)
+                    // RDNA matrix C is column-major.
+                    mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
+#else
+                    // swap A and B for CUDA.
                     mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[0], K_A);
+#endif // defined(AMD_WMMA_AVAILABLE)
                 }
             }
         }
@@ -532,7 +581,13 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
 #pragma unroll
             for (int l = 0; l < T_C_KQ::ne; ++l) {
                 if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
-                    KQ_max_new[l % 2] = fmaxf(KQ_max_new[l % 2], KQ_C[k0/(np*T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
+#if defined(AMD_WMMA_AVAILABLE)
+                    constexpr int KQ_idx = 0;
+#else
+                    // Turing + Volta:
+                    const int KQ_idx = l % 2;
+#endif // defined(AMD_WMMA_AVAILABLE)
+                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/(np*T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
                 }
             }
         }
@@ -552,8 +607,14 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
 #pragma unroll
             for (int l = 0; l < T_C_KQ::ne; ++l) {
                 if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
-                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = expf(KQ_C[k0/(np*T_C_KQ::I)].x[l] - KQ_max_new[l % 2]);
-                    KQ_rowsum_add[l % 2] += KQ_C[k0/(np*T_C_KQ::I)].x[l];
+#if defined(AMD_WMMA_AVAILABLE)
+                    constexpr int KQ_idx = 0;
+#else
+                    // Turing + Volta:
+                    const int KQ_idx = l % 2;
+#endif // defined(AMD_WMMA_AVAILABLE)
+                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = expf(KQ_C[k0/(np*T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
+                    KQ_rowsum_add[KQ_idx] += KQ_C[k0/(np*T_C_KQ::I)].x[l];
                 } else {
                     KQ_C[k0/(np*T_C_KQ::I)].x[l] = 0.0f;
                 }
@@ -584,8 +645,13 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
 #pragma unroll
             for (int l = 0; l < T_C_KQ::ne; ++l) {
                 if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
+#if defined(AMD_WMMA_AVAILABLE)
+                    constexpr int KQ_idx = 0;
+#else
                     // Turing + Volta:
-                    KQ_max_new[(l/2) % 2] = fmaxf(KQ_max_new[(l/2) % 2], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
+                    const int KQ_idx = (l/2) % 2;
+#endif // defined(AMD_WMMA_AVAILABLE)
+                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
                 }
             }
         }
@@ -596,7 +662,11 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
             // Values per KQ column are spread across 4 threads:
             constexpr int offset_first = 2;
             constexpr int offset_last  = 1;
-#else
+#elif defined(AMD_WMMA_AVAILABLE)
+            // Values per KQ column are spread across 2 threads:
+            constexpr int offset_first = 16;
+            constexpr int offset_last  = 16;
+#else // Volta
             // Values per KQ column are spread across 2 threads:
             constexpr int offset_first = 2;
             constexpr int offset_last  = 2;
@@ -612,10 +682,15 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
         for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
 #pragma unroll
             for (int l = 0; l < T_C_KQ::ne; ++l) {
-                // Turing + Volta:
                 if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
-                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[(l/2) % 2]);
-                    KQ_rowsum_add[(l/2) % 2] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
+#if defined(AMD_WMMA_AVAILABLE)
+                    constexpr int KQ_idx = 0;
+#else
+                    // Turing + Volta:
+                    const int KQ_idx = (l/2) % 2;
+#endif // defined(AMD_WMMA_AVAILABLE)
+                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[KQ_idx]);
+                    KQ_rowsum_add[KQ_idx] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
                 } else {
                     KQ_C[(k0/(np*T_C_KQ::J))].x[l] = 0.0f;
                 }
@@ -639,7 +714,7 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
 
 #if defined(TURING_MMA_AVAILABLE)
         if constexpr (cols_per_warp == 8) {
-            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
+            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
 #pragma unroll
             for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
 #pragma unroll
@@ -660,6 +735,16 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
                 }
             }
         }
+#elif defined(AMD_WMMA_AVAILABLE)
+        const half2 KQ_max_scale_h2 = make_half2(
+            KQ_max_scale[0], KQ_max_scale[0]);
+#pragma unroll
+        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
+#pragma unroll
+            for (int l = 0; l < T_C_VKQ::ne; ++l) {
+                VKQ_C[i].x[l] *= KQ_max_scale_h2;
+            }
+        }
 #else // Volta
         const half2 KQ_max_scale_h2 = make_half2(
             KQ_max_scale[(threadIdx.x / 2) % 2], KQ_max_scale[(threadIdx.x / 2) % 2]);
@@ -707,6 +792,10 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
     // Therefore, iterate over V in reverse and re-use the data if possible.
     static_assert(!mla || nstages <= 1, "combination of MLA and multi-stage loading not implemented");
     constexpr int reusable_cutoff = mla ? (DKQ - 1) - (DKQ - 1) % (2*nbatch_K2) - (DKQ - DV) : DV;
+#if defined(AMD_WMMA_AVAILABLE) && !defined(LDMATRIX_TRANS_AVAILABLE)
+    T_A_VKQ A_identity;
+    make_identity_mat(A_identity);
+#endif // defined(AMD_WMMA_AVAILABLE) && !defined(LDMATRIX_TRANS_AVAILABLE)
 
     // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
 #pragma unroll
@@ -727,7 +816,7 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
         }
         const half2 * tile_V_i = i0_start < reusable_cutoff ? tile_V : tile_V + (i0_start - reusable_cutoff)/2;
 
-#if defined(TURING_MMA_AVAILABLE)
+#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
         constexpr int i0_stride = cols_per_warp == 8 ? T_C_VKQ::I : 2*T_C_VKQ::J;
 #pragma unroll
         for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
@@ -737,12 +826,26 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
                 const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;
 
                 T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
+#if defined(LDMATRIX_TRANS_AVAILABLE)
                 load_ldmatrix_trans(A, tile_V_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
+#else
+                // TODO: Try to transpose tile_V when loading gmem to smem.
+                // Use mma to transpose T_A_VKQ for RDNA.
+                T_A_VKQ A_trans;
+                load_ldmatrix(A_trans, tile_V_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
+                mma(A, A_trans, A_identity);
+#endif // defined(TURING_MMA_AVAILABLE)
                 if constexpr (T_B_KQ::I == 8) {
                     mma(VKQ_C[i_VKQ_0/i0_stride], A, B[k00/(np*T_A_VKQ::J)]);
                 } else {
-                    // Wide version of VKQ_C is column-major => swap A and B.
+                    // Wide version of VKQ_C is column-major.
+#if defined(AMD_WMMA_AVAILABLE)
+                    // RDNA matrix C is column-major.
+                    mma(VKQ_C[i_VKQ_0/i0_stride], A, B[k00/(np*T_A_VKQ::J)]);
+#else
+                    // swap A and B for CUDA.
                     mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::J)], A);
+#endif // defined(AMD_WMMA_AVAILABLE)
                 }
             }
         }
@@ -761,7 +864,7 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
                 mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::I)], A);
             }
         }
-#endif // defined(TURING_MMA_AVAILABLE)
+#endif // defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
 
         if constexpr (nstages <= 1) {
             __syncthreads(); // Only needed if tile_K == tile_V.
@@ -774,7 +877,7 @@ static __device__ __forceinline__ void flash_attn_ext_f16_iter(
         tile_Q, tile_K, tile_V, tile_mask,
         Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0);
     NO_DEVICE_CODE;
-#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
+#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4))
 }
 
 #if defined(TURING_MMA_AVAILABLE)
@@ -794,6 +897,15 @@ template<> struct mma_tile_sizes<8> {
     using T_B_VKQ = tile< 8,  8, half2>; // column-major
     using T_C_VKQ = tile<16,  4, half2>; // row-major
 };
+#elif defined(AMD_WMMA_AVAILABLE)
+template<int ncols> struct mma_tile_sizes {
+    using T_A_KQ  = tile<16,  8, half2>; // row-major
+    using T_B_KQ  = tile<16,  8, half2>; // column-major
+    using T_C_KQ  = tile<16, 16, float>; // column-major
+    using T_A_VKQ = tile<16,  8, half2>; // row-major
+    using T_B_VKQ = tile<16,  8, half2>; // column-major
+    using T_C_VKQ = tile<16,  8, half2>; // column-major
+};
 #else // Volta
 template<int ncols> struct mma_tile_sizes {
     using T_A_KQ  = tile< 8,  4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
@@ -828,7 +940,7 @@ static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
         const int jt,
         const int kb0_start,
         const int kb0_stop) {
-#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
+#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4))
     //In this kernel Q, K, V are matrices while i, j, k are matrix indices.
 
     constexpr int ncols = ncols1 * ncols2;
@@ -840,7 +952,7 @@ static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
     using     T_C_VKQ   = typename mma_tile_sizes<ncols>::T_C_VKQ;
 
     constexpr int  cols_per_warp   = T_B_KQ::I;
-    constexpr int  cols_per_thread = 2; // This is specifically KQ columns, Volta only has a single VKQ column.
+    constexpr int  cols_per_thread = get_cols_per_thread();
     constexpr int  np              = nwarps * (cols_per_warp/ncols2) / ncols1; // Number of parallel CUDA warps per Q column.
     constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
     constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols);
@@ -871,6 +983,8 @@ static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
     T_B_KQ    Q_B[(Q_in_reg ? DKQ/(2*T_B_KQ::J) : 1)];
 #if defined(TURING_MMA_AVAILABLE)
     T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)];
+#elif defined(AMD_WMMA_AVAILABLE)
+    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
 #else // Volta
     T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
 #endif // defined(TURING_MMA_AVAILABLE)
@@ -1010,6 +1124,10 @@ static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
         // The partial sums are spread across 8/4 threads.
         constexpr int offset_first = cols_per_warp == 8 ? 16 : 2;
         constexpr int offset_last  = cols_per_warp == 8 ?  4 : 1;
+#elif defined(AMD_WMMA_AVAILABLE)
+        // The partial sums are spread across 2 threads.
+        constexpr int offset_first = 16;
+        constexpr int offset_last  = 16;
 #else // Volta
         // The partial sums are spread across 2 threads.
         constexpr int offset_first = 2;
@@ -1047,7 +1165,7 @@ static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
 
 #if defined(TURING_MMA_AVAILABLE)
         if constexpr (cols_per_warp == 8) {
-            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
+            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
 #pragma unroll
             for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
 #pragma unroll
@@ -1068,6 +1186,15 @@ static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
                 }
             }
         }
+#elif defined(AMD_WMMA_AVAILABLE)
+        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[0]);
+#pragma unroll
+        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
+#pragma unroll
+            for (int l = 0; l < T_C_VKQ::ne; ++l) {
+                VKQ_C[i].x[l] *= KQ_max_scale_h2;
+            }
+        }
 #else // Volta
         const int col = (threadIdx.x / 2) % 2;
         const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
@@ -1119,6 +1246,10 @@ static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
         const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
         const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
         const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;
+#elif defined(AMD_WMMA_AVAILABLE)
+        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(0);
+        const float2 KQ_cmr = make_float2(KQ_max[0], KQ_rowsum[0]);
+        const bool thread_should_write = threadIdx.x / 16 < cols_per_thread;
 #else // Volta
         const int jc_cwm = threadIdx.y*cols_per_warp + T_C_KQ::get_i(threadIdx.x & 2);
         const float2 KQ_cmr = make_float2(KQ_max[(threadIdx.x & 2) / 2], KQ_rowsum[(threadIdx.x & 2) / 2]);
@@ -1319,7 +1450,7 @@ static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
         stride_Q1, stride_Q2, stride_K, stride_V, stride_mask,
         jt, kb0_start, kb0_stop);
     NO_DEVICE_CODE;
-#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
+#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4))
 }
 
 template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool mla>
@@ -1346,7 +1477,7 @@ static __global__ void flash_attn_ext_f16(
                             const int32_t nb21, const int32_t nb22, const int64_t nb23,
                             const int32_t ne31, const int32_t ne32, const int32_t ne33,
                             const int32_t nb31, const int32_t nb32, const int64_t nb33) {
-#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE))
+#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)))
 
     // Skip unused kernel variants for faster compilation:
     if (use_logit_softcap && !(DKQ == 128 || DKQ == 256)) {
@@ -1360,6 +1491,13 @@ static __global__ void flash_attn_ext_f16(
     }
 #endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING
 
+#if defined(AMD_WMMA_AVAILABLE)
+    if (ncols1*ncols2 > 32 || ncols1*ncols2 < 16 || DKQ > 128 || ncols2 == 1) {
+        NO_DEVICE_CODE;
+        return;
+    }
+#endif // defined(AMD_WMMA_AVAILABLE)
+
     static_assert(!mla || DKQ >= DV, "MLA needs DKQ >= DV");
 
     constexpr int ncols     = ncols1 * ncols2;
@@ -1473,7 +1611,7 @@ static __global__ void flash_attn_ext_f16(
               ne31, ne32, ne33,
               nb31, nb32, nb33);
     NO_DEVICE_CODE;
-#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE))
+#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)))
 }
 
 template <int DKQ, int DV, int ncols1, int ncols2>
@@ -1492,7 +1630,7 @@ void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml
     const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);
     const int  nstages        = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, cc);
 
-    const int cols_per_warp = std::min(ncols, turing_mma_available(cc) ? 16 : 32);
+    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
     const int nwarps        = nthreads / WARP_SIZE;
 
     constexpr bool mla = DKQ == 576;
@@ -1512,29 +1650,34 @@ void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml
     float logit_softcap;
     memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
 
+#if defined(GGML_USE_HIP)
+    using fattn_kernel_ptr_t = const void*;
+#else
+    using fattn_kernel_ptr_t = fattn_kernel_t;
+#endif // defined(GGML_USE_HIP)
     fattn_kernel_t fattn_kernel;
     if (logit_softcap == 0.0f) {
         constexpr bool use_logit_softcap = false;
         fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, mla>;
 
-#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
+#if !defined(GGML_USE_MUSA)
         static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
         if (!shared_memory_limit_raised[id]) {
-            CUDA_CHECK(cudaFuncSetAttribute(fattn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
+            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
             shared_memory_limit_raised[id] = true;
         }
-#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
+#endif // !defined(GGML_USE_MUSA)
     } else {
         constexpr bool use_logit_softcap = true;
         fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, mla>;
 
-#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
+#if !defined(GGML_USE_MUSA)
         static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
         if (!shared_memory_limit_raised[id]) {
-            CUDA_CHECK(cudaFuncSetAttribute(fattn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
+            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
             shared_memory_limit_raised[id] = true;
         }
-#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
+#endif // !defined(GGML_USE_MUSA)
     }
 
     launch_fattn<DV, ncols1, ncols2>
diff --git a/ggml/src/ggml-cuda/fattn.cu b/ggml/src/ggml-cuda/fattn.cu
index 015540666..be21c6553 100644
--- a/ggml/src/ggml-cuda/fattn.cu
+++ b/ggml/src/ggml-cuda/fattn.cu
@@ -6,6 +6,11 @@
 #include "fattn-wmma-f16.cuh"
 #include "fattn.cuh"
 
+// GFX906 Q8 Flash Attention kernel
+#ifdef GGML_USE_HIP
+    #include "gfx906/gfx906-fattn-q8.cuh"
+#endif
+
 template <int DKQ, int DV, int ncols2>
 static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
@@ -18,12 +23,12 @@ static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_con
         }
     }
 
-    if (turing_mma_available(cc) && Q->ne[1] <= 16/ncols2) {
+    if ((turing_mma_available(cc) || amd_wmma_available(cc)) && Q->ne[1] <= 16/ncols2) {
         ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
         return;
     }
 
-    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || Q->ne[1] <= 32/ncols2) {
+    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || amd_wmma_available(cc) || Q->ne[1] <= 32/ncols2) {
         ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
         return;
     }
@@ -208,6 +213,9 @@ enum best_fattn_kernel {
     BEST_FATTN_KERNEL_VEC      = 100,
     BEST_FATTN_KERNEL_WMMA_F16 = 300,
     BEST_FATTN_KERNEL_MMA_F16  = 400,
+#ifdef GGML_USE_HIP
+    BEST_FATTN_KERNEL_TILE_Q8  = 250,
+#endif
 };
 
 static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
@@ -230,7 +238,18 @@ static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const
 
     // The effective batch size for the kernel can be increased by gqa_ratio.
     // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
-    const bool gqa_opt_applies = gqa_ratio % 2 == 0 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
+    bool gqa_opt_applies = gqa_ratio % 2 == 0 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
+    for (const ggml_tensor * t : {Q, K, V, mask}) {
+        if (t == nullptr) {
+            continue;
+        }
+        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
+            if (t->nb[i] % 16 != 0) {
+                gqa_opt_applies = false;
+                break;
+            }
+        }
+    }
 
     const int cc = ggml_cuda_info().devices[device].cc;
 
@@ -329,6 +348,21 @@ static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const
         return BEST_FATTN_KERNEL_MMA_F16;
     }
 
+    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
+        int gqa_ratio_eff = 1;
+        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
+        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
+            gqa_ratio_eff *= 2;
+        }
+        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
+            return BEST_FATTN_KERNEL_VEC;
+        }
+        if (Q->ne[1] * gqa_ratio_eff <= 16) {
+            return BEST_FATTN_KERNEL_TILE;
+        }
+        return BEST_FATTN_KERNEL_MMA_F16;
+    }
+
     // Use the WMMA kernel if possible:
     if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 576) {
         if (can_use_vector_kernel && Q->ne[1] <= 2) {
@@ -337,6 +371,31 @@ static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const
         return BEST_FATTN_KERNEL_WMMA_F16;
     }
 
+    if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && Q->ne[0] <= 128 && Q->ne[0] != 40 && Q->ne[0] != 72) {
+        if (can_use_vector_kernel) {
+            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
+                if (Q->ne[1] == 1) {
+                    if (!gqa_opt_applies) {
+                        return BEST_FATTN_KERNEL_VEC;
+                    }
+                }
+            } else {
+                if (Q->ne[1] <= 2) {
+                    return BEST_FATTN_KERNEL_VEC;
+                }
+            }
+        }
+        int gqa_ratio_eff = 1;
+        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
+        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
+            gqa_ratio_eff *= 2;
+        }
+        if (Q->ne[1] * gqa_ratio_eff <= 8) {
+            return BEST_FATTN_KERNEL_TILE; // AMD WMMA is only faster if the full tile width of 16 can be utilized.
+        }
+        return BEST_FATTN_KERNEL_MMA_F16;
+    }
+
     // If there are no tensor cores available, use the generic tile kernel:
     if (can_use_vector_kernel) {
         if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
@@ -345,23 +404,51 @@ static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const
                     return BEST_FATTN_KERNEL_VEC;
                 }
             }
-        } else {
+        }
+#ifndef GGML_USE_HIP
+        else {
             if (Q->ne[1] <= 2) {
                 return BEST_FATTN_KERNEL_VEC;
             }
         }
+#endif
     }
+
+#ifdef GGML_USE_HIP
+    if (K->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q8_0) {
+        const bool q8_head_size_supported = (K->ne[0] % 32 == 0) &&
+                                            (K->ne[0] != 40) &&
+                                            (K->ne[0] != 80) &&
+                                            (K->ne[0] != 112);
+
+        if (q8_head_size_supported) {
+            const char * env_use_dot4 = getenv("GGML_HIP_FATTN_USE_TILE_DOT4");
+            if (env_use_dot4 == nullptr || strcmp(env_use_dot4, "0") != 0) {
+                return BEST_FATTN_KERNEL_TILE_Q8;
+            }
+        }
+    }
+#endif
+
     return BEST_FATTN_KERNEL_TILE;
 }
 
 void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     ggml_cuda_set_device(ctx.device);
-    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
+
+    best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst);
+
+    switch (kernel) {
         case BEST_FATTN_KERNEL_NONE:
             GGML_ABORT("fatal error");
         case BEST_FATTN_KERNEL_TILE:
             ggml_cuda_flash_attn_ext_tile(ctx, dst);
             break;
+#ifdef GGML_USE_HIP
+        case BEST_FATTN_KERNEL_TILE_Q8:
+            ggml_cuda_flash_attn_ext_tile_q8(ctx, dst);
+            break;
+#endif // GGML_USE_HIP
         case BEST_FATTN_KERNEL_VEC:
             ggml_cuda_flash_attn_ext_vec(ctx, dst);
             break;
diff --git a/ggml/src/ggml-cuda/gfx906/gfx906-common.cuh b/ggml/src/ggml-cuda/gfx906/gfx906-common.cuh
new file mode 100644
index 000000000..bda67ee62
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/gfx906-common.cuh
@@ -0,0 +1,242 @@
+#pragma once
+
+#include "gfx906-config.h"
+
+#ifdef GGML_USE_HIP
+
+static __device__ __forceinline__ float sgpr_broadcast_f32(float value) {
+    int i = __float_as_int(value);
+    i = __builtin_amdgcn_readfirstlane(i);
+    return __int_as_float(i);
+}
+
+static __device__ __forceinline__ int sgpr_broadcast_i32(int value) {
+    return __builtin_amdgcn_readfirstlane(value);
+}
+
+static __device__ __forceinline__ half sgpr_broadcast_f16(half value) {
+    int i = *reinterpret_cast<const short*>(&value);
+    i = __builtin_amdgcn_readfirstlane(i);
+    short s = static_cast<short>(i);
+    return *reinterpret_cast<half*>(&s);
+}
+
+static __device__ __forceinline__ float fast_exp_f32(float x) {
+    constexpr float LOG2_E = 1.4426950408889634f;
+    float result;
+    asm volatile(
+        "v_exp_f32 %0, %1"
+        : "=v"(result)
+        : "v"(x * LOG2_E)
+    );
+    return result;
+}
+
+static __device__ __forceinline__ float fast_exp2_f32(float x) {
+    float result;
+    asm volatile(
+        "v_exp_f32 %0, %1"
+        : "=v"(result)
+        : "v"(x)
+    );
+    return result;
+}
+
+static __device__ __forceinline__ float fast_log2_f32(float x) {
+    float result;
+    asm volatile(
+        "v_log_f32 %0, %1"
+        : "=v"(result)
+        : "v"(x)
+    );
+    return result;
+}
+
+static __device__ __forceinline__ float fast_tanh_f32(float x) {
+    if (x > 10.0f) return 1.0f;
+    if (x < -10.0f) return -1.0f;
+
+    const float exp2x = fast_exp_f32(2.0f * x);
+    return 1.0f - 2.0f / (exp2x + 1.0f);
+}
+
+static __device__ __forceinline__ float fast_rcp_f32(float x) {
+    float result;
+    asm volatile(
+        "v_rcp_f32 %0, %1"
+        : "=v"(result)
+        : "v"(x)
+    );
+    return result;
+}
+
+#define DEFINE_FUSED_DPP_F32(name, barrier, dpp_ctrl, vop_instr)           \
+    static __device__ __forceinline__ float name(float x) {                \
+        float result;                                                       \
+        asm volatile(                                                       \
+            barrier                                                         \
+            vop_instr " %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf" \
+            : "=v"(result) : "v"(x) : "memory"                             \
+        );                                                                  \
+        return result;                                                      \
+    }
+
+DEFINE_FUSED_DPP_F32(hip_add_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_add_f32_dpp")
+DEFINE_FUSED_DPP_F32(hip_max_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_max_f32_dpp")
+
+DEFINE_FUSED_DPP_F32(hip_add_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_add_f32_dpp")
+DEFINE_FUSED_DPP_F32(hip_max_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_max_f32_dpp")
+
+DEFINE_FUSED_DPP_F32(hip_add_xor8_f32, "s_nop 1\n", "row_ror:8", "v_add_f32_dpp")
+DEFINE_FUSED_DPP_F32(hip_max_xor8_f32, "s_nop 1\n", "row_ror:8", "v_max_f32_dpp")
+
+#undef DEFINE_FUSED_DPP_F32
+
+static __device__ __forceinline__ float hip_shuffle_xor4_f32(float x) {
+    int v_src = __float_as_int(x);
+    int v_dst;
+    asm volatile(
+        "v_mov_b32 %0, %1\n"
+        "s_nop 1\n"
+        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
+        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
+        : "=v"(v_dst) : "v"(v_src) : "memory"
+    );
+    return __int_as_float(v_dst);
+}
+
+static __device__ __forceinline__ float hip_shuffle_xor16_f32(float x) {
+    int int_val = __float_as_int(x);
+    int result;
+    asm volatile(
+        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
+        "s_waitcnt lgkmcnt(0)\n"
+        : "=v"(result) : "v"(int_val) : "memory"
+    );
+    return __int_as_float(result);
+}
+
+struct AddOp {
+    static __device__ __forceinline__ float apply(float a, float b) { return a + b; }
+    static __device__ __forceinline__ float xor1(float x) { return hip_add_xor1_f32(x); }
+    static __device__ __forceinline__ float xor2(float x) { return hip_add_xor2_f32(x); }
+    static __device__ __forceinline__ float xor8(float x) { return hip_add_xor8_f32(x); }
+};
+
+struct MaxOp {
+    static __device__ __forceinline__ float apply(float a, float b) { return fmaxf(a, b); }
+    static __device__ __forceinline__ float xor1(float x) { return hip_max_xor1_f32(x); }
+    static __device__ __forceinline__ float xor2(float x) { return hip_max_xor2_f32(x); }
+    static __device__ __forceinline__ float xor8(float x) { return hip_max_xor8_f32(x); }
+};
+
+template<int width = WARP_SIZE, typename Op>
+static __device__ __forceinline__ float warp_reduce_amd_f32(float x) {
+    if (width >= 2)  x = Op::xor1(x);
+    if (width >= 4)  x = Op::xor2(x);
+    if (width >= 8)  x = Op::apply(x, hip_shuffle_xor4_f32(x));
+    if (width >= 16) x = Op::xor8(x);
+    if (width >= 32) x = Op::apply(x, hip_shuffle_xor16_f32(x));
+    if (width == 64) x = Op::apply(x, __shfl_xor(x, 32, 64));
+    return x;
+}
+
+template<typename T>
+static __device__ __forceinline__ T hip_dpp_xor1(T value) {
+    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
+    int int_val = *reinterpret_cast<int*>(&value);
+    int result;
+    asm volatile(
+        "s_nop 4\n"
+        "v_mov_b32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
+        : "=v"(result) : "v"(int_val) : "memory"
+    );
+    return *reinterpret_cast<T*>(&result);
+}
+
+template<typename T>
+static __device__ __forceinline__ T hip_dpp_xor2(T value) {
+    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
+    int int_val = *reinterpret_cast<int*>(&value);
+    int result;
+    asm volatile(
+        "s_nop 1\n"
+        "v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf"
+        : "=v"(result) : "v"(int_val) : "memory"
+    );
+    return *reinterpret_cast<T*>(&result);
+}
+
+template<typename T>
+static __device__ __forceinline__ T hip_dpp_xor4(T value) {
+    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
+    int v_src = *reinterpret_cast<int*>(&value);
+    int v_dst;
+    asm volatile(
+        "v_mov_b32 %0, %1\n"
+        "s_nop 1\n"
+        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
+        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
+        : "=v"(v_dst) : "v"(v_src) : "memory"
+    );
+    return *reinterpret_cast<T*>(&v_dst);
+}
+
+template<typename T>
+static __device__ __forceinline__ T hip_dpp_xor8(T value) {
+    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
+    int int_val = *reinterpret_cast<int*>(&value);
+    int result;
+    asm volatile(
+        "s_nop 1\n"
+        "v_mov_b32_dpp %0, %1 row_ror:8 row_mask:0xf bank_mask:0xf"
+        : "=v"(result) : "v"(int_val) : "memory"
+    );
+    return *reinterpret_cast<T*>(&result);
+}
+
+template<typename T>
+static __device__ __forceinline__ T hip_dpp_xor16(T value) {
+    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
+    int int_val = *reinterpret_cast<int*>(&value);
+    int result;
+    asm volatile(
+        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
+        "s_waitcnt lgkmcnt(0)\n"
+        : "=v"(result) : "v"(int_val) : "memory"
+    );
+    return *reinterpret_cast<T*>(&result);
+}
+
+template<int width = WARP_SIZE, typename T>
+static __device__ __forceinline__ T gfx906_shfl_xor_sync(T x, int offset) {
+    switch (~offset) {
+        case ~1:  return hip_dpp_xor1(x);
+        case ~2:  return hip_dpp_xor2(x);
+        case ~4:  return hip_dpp_xor4(x);
+        case ~8:  return hip_dpp_xor8(x);
+        case ~16: return hip_dpp_xor16(x);
+        default:  return __shfl_xor(x, offset, width);
+    }
+}
+
+template<int width = WARP_SIZE>
+static __device__ __forceinline__ float gfx906_warp_reduce_sum_f32(float x) {
+    return warp_reduce_amd_f32<width, AddOp>(x);
+}
+
+template<int width = WARP_SIZE>
+static __device__ __forceinline__ float gfx906_warp_reduce_max_f32(float x) {
+    return warp_reduce_amd_f32<width, MaxOp>(x);
+}
+
+template<int width = WARP_SIZE, typename T>
+static __device__ __forceinline__ T gfx906_warp_reduce_sum_generic(T x) {
+    #pragma unroll
+    for (int offset = width/2; offset > 0; offset >>= 1) {
+        x += gfx906_shfl_xor_sync<width>(x, offset);
+    }
+    return x;
+}
+
+#endif
diff --git a/ggml/src/ggml-cuda/gfx906/gfx906-config.h b/ggml/src/ggml-cuda/gfx906/gfx906-config.h
new file mode 100644
index 000000000..c20be3cd5
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/gfx906-config.h
@@ -0,0 +1,26 @@
+#pragma once
+
+// GFX906 (Vega 20 / MI50) kernel configuration
+
+#ifdef GGML_USE_HIP
+
+#define GFX906_FATTN_SPLIT_K_ENABLED 0
+
+#if GFX906_FATTN_SPLIT_K_ENABLED
+    #define GFX906_FATTN_N_SPLIT_MAX 8
+#else
+    #define GFX906_FATTN_N_SPLIT_MAX 1
+#endif
+
+#define GFX906_MMQ_ITER_K 256
+#define GFX906_MMQ_NWARPS 2
+
+#define GFX906_FATTN_Q8_ENABLED 1
+#define GFX906_Q8_SUPPORTS_HEAD_DIM(d) \
+    ((d) % 32 == 0 && (d) != 40 && (d) != 80 && (d) != 112)
+
+#define GFX906_USE_DPP_REDUCTIONS 1
+#define GFX906_FATTN_TILE_SIZE_DEFAULT 128
+#define GFX906_Q8_SCALE_HOISTING 1
+
+#endif // GGML_USE_HIP
diff --git a/ggml/src/ggml-cuda/gfx906/gfx906-fattn-q8.cu b/ggml/src/ggml-cuda/gfx906/gfx906-fattn-q8.cu
new file mode 100644
index 000000000..49c61d465
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/gfx906-fattn-q8.cu
@@ -0,0 +1,34 @@
+// Q8 Flash Attention dispatch: head sizes 64, 96, 128, 256, 576
+
+#include "../common.cuh"
+#include "gfx906-fattn-q8.cuh"
+
+void ggml_cuda_flash_attn_ext_tile_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
+    const ggml_tensor * K = dst->src[1];
+    const ggml_tensor * V = dst->src[2];
+    switch (K->ne[0]) {
+        case  64: {
+            GGML_ASSERT(V->ne[0] == K->ne[0]);
+            ggml_cuda_flash_attn_ext_tile_q8_case< 64,  64>(ctx, dst);
+        } break;
+        case  96: {
+            GGML_ASSERT(V->ne[0] == K->ne[0]);
+            ggml_cuda_flash_attn_ext_tile_q8_case< 96,  96>(ctx, dst);
+        } break;
+        case 128: {
+            GGML_ASSERT(V->ne[0] == K->ne[0]);
+            ggml_cuda_flash_attn_ext_tile_q8_case<128, 128>(ctx, dst);
+        } break;
+        case 256: {
+            GGML_ASSERT(V->ne[0] == K->ne[0]);
+            ggml_cuda_flash_attn_ext_tile_q8_case<256, 256>(ctx, dst);
+        } break;
+        case 576: {
+            GGML_ASSERT(V->ne[0] == 512);
+            ggml_cuda_flash_attn_ext_tile_q8_case<576, 512>(ctx, dst);
+        } break;
+        default: {
+            GGML_ABORT("Unsupported head size for Q8 tile kernel");
+        } break;
+    }
+}
diff --git a/ggml/src/ggml-cuda/gfx906/gfx906-fattn-q8.cuh b/ggml/src/ggml-cuda/gfx906/gfx906-fattn-q8.cuh
new file mode 100644
index 000000000..da547eb3e
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/gfx906-fattn-q8.cuh
@@ -0,0 +1,978 @@
+// Flash Attention with Q8_0 quantized KV cache for GFX906
+// Uses v_dot4_i32_i8 for INT8 dot products
+
+#include "../common.cuh"
+#include "../fattn-common.cuh"
+#include "../cpy-utils.cuh"
+
+#include "gfx906-config.h"
+#include "gfx906-common.cuh"
+
+#define GGML_CUDA_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
+    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
+        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
+        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
+        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
+        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
+        return ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
+    }
+
+static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_q8_get_config_amd(const int DKQ, const int DV, const int ncols) {
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  32,  40)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  32,  40)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  32,  40)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  32,  40)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  32,  40)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 64, 256, 2,  32,  40)
+
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2,  64, 3,  32,  64)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 3,  64,  64)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 128, 2,  32,  64)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 256, 2, 128,  64)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 64, 256, 2,  64,  64)
+
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  32,  40)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  32,  40)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  32,  40)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  32,  40)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  32,  40)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 64, 256, 2,  32,  40)
+
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  32,  48)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  32,  48)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  32,  48)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  32,  48)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  32,  48)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 64, 256, 2,  32,  48)
+
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  32,  56)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  32,  56)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  32,  56)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32,  56)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  32,  56)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 64, 256, 2,  32,  56)
+
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2, 256, 2, 128,  64)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 2,  64, 128)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 256, 2,  64, 128)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 2,  64, 128)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 64, 256, 2,  64,  64)
+
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2, 256, 2, 128,  64)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 256, 2,  64, 128)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  64, 128)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  32, 128)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  32, 128)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  64,  64)
+    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 32, 512, 1, 128,  64)
+
+    return 0;
+}
+
+static __host__ uint32_t ggml_cuda_fattn_tile_q8_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
+    return ggml_cuda_fattn_tile_q8_get_config_amd(DKQ, DV, ncols);
+}
+
+static constexpr __device__ uint32_t ggml_cuda_fattn_tile_q8_get_config(const int DKQ, const int DV, const int ncols) {
+    return ggml_cuda_fattn_tile_q8_get_config_amd(DKQ, DV, ncols);
+}
+
+static __host__ int ggml_cuda_fattn_tile_q8_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc) {
+    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols, cc) >> 0) & ((1 << 10) - 1);
+}
+
+static constexpr __device__ int ggml_cuda_fattn_tile_q8_get_nthreads(const int DKQ, const int DV, const int ncols) {
+    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
+}
+
+static __host__ int ggml_cuda_fattn_tile_q8_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc) {
+    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols, cc) >> 10) & ((1 << 4) - 1);
+}
+
+static constexpr __device__ int ggml_cuda_fattn_tile_q8_get_occupancy(const int DKQ, const int DV, const int ncols) {
+    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
+}
+
+static __host__ int ggml_cuda_fattn_tile_q8_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc) {
+    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols, cc) >> 14) & ((1 << 9) - 1);
+}
+
+static constexpr __device__ int ggml_cuda_fattn_tile_q8_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
+    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
+}
+
+static __host__ int ggml_cuda_fattn_tile_q8_get_nbatch_K(const int DKQ, const int DV, const int ncols, const int cc) {
+    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols, cc) >> 23) & ((1 << 9) - 1);
+}
+
+static constexpr __device__ int ggml_cuda_fattn_tile_q8_get_nbatch_K(const int DKQ, const int DV, const int ncols) {
+    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
+}
+
+template<int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
+static __device__ __forceinline__ void flash_attn_tile_q8_q8_load_tile(
+        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int stride_KV, const int i_sup) {
+    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
+    constexpr int cpy_ne = cpy_nb / 4;
+
+    auto load = [&] __device__ (const int n) {
+        const int stride_j = warp_size >> n;
+
+        if (stride_j == 0) {
+            return;
+        }
+
+        const int j0_start = stride_j == warp_size ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
+        const int j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
+        const int stride_i = warp_size / stride_j;
+
+        if (j0_start == j0_stop) {
+            return;
+        }
+
+#pragma unroll
+        for (int i0 = 0; i0 < I; i0 += nwarps*stride_i) {
+            const int i = i0 + threadIdx.y*stride_i + (stride_j == warp_size ? 0 : threadIdx.x / stride_j);
+
+            if (i0 + nwarps*stride_i <= I || i < I) {
+#pragma unroll
+                for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
+                    const int j = j0*cpy_ne + (stride_j == warp_size ? threadIdx.x : threadIdx.x % stride_j)*cpy_ne;
+
+                    const half2 zero[cpy_ne] = {{0.0f, 0.0f}};
+                    ggml_cuda_memcpy_1<cpy_nb>(
+                        tile_KV + i*(J/2 + J_padding) + j,
+                        !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);
+                }
+            }
+        }
+    };
+
+    static_assert(J % 8 == 0, "bad J");
+    static_assert((J/2) % cpy_ne == 0, "bad J");
+    ggml_cuda_unroll<7>{}(load);
+}
+
+template<int warp_size, int nwarps, int I, int J, int K_row_stride, bool oob_check>
+static __device__ __forceinline__ void flash_attn_tile_q8_q8_load_tile_q8(
+        const block_q8_0 * const __restrict__ K_q8,
+        int8_t * const __restrict__ K_values,
+        half * const __restrict__ K_scales,
+        const int stride_K_q8,
+        const int i_sup) {
+
+    if constexpr (J > 0) {
+        constexpr int blocks_per_row = J / 32;
+        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
+        const int total_blocks = I * blocks_per_row;
+
+        for (int block_idx = tid; block_idx < total_blocks; block_idx += blockDim.x * blockDim.y) {
+            const int row = block_idx / blocks_per_row;
+            const int col_block = block_idx % blocks_per_row;
+
+            if (oob_check && row >= i_sup) {
+                break;
+            }
+
+            const int global_block_idx = row * stride_K_q8 + col_block;
+            const block_q8_0 src_block = K_q8[global_block_idx];
+
+            K_scales[col_block * I + row] = src_block.d;
+
+            int8_t * dst = K_values + row * K_row_stride + col_block * 32;
+            const int4* src_int4 = (const int4*)src_block.qs;
+            int4* dst_int4 = (int4*)dst;
+
+            dst_int4[0] = src_int4[0];
+            dst_int4[1] = src_int4[1];
+        }
+
+        __syncthreads();
+    }
+}
+
+template<int nthreads, int ncols, int ncols2, int DKQ>
+static __device__ __forceinline__ void flash_attn_tile_q8_quantize_Q_to_shared(
+        const float * __restrict__ Q_f,
+        int8_t * __restrict__ Q_values,
+        half * __restrict__ Q_scales,
+        const int col_Q_0,
+        const int ne01,
+        const int32_t nb01,
+        const int32_t nb02,
+        const float scale) {
+
+    constexpr int blocks_per_col = DKQ / QK8_0;
+    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
+    const int total_blocks = ncols * blocks_per_col;
+
+    for (int block_idx = tid; block_idx < total_blocks; block_idx += blockDim.x * blockDim.y) {
+        const int col = block_idx / blocks_per_col;
+        const int col_block = block_idx % blocks_per_col;
+        const int block_start = col_block * QK8_0;
+
+        const int jc = col;
+        const int j = jc / ncols2;
+        const int c = jc % ncols2;
+
+        if (ncols != 1 && col_Q_0 + j >= ne01) {
+            continue;
+        }
+
+        float Q_vals[QK8_0];
+        block_q8_0 Q_block;
+
+        const int base_offset = c*(nb02/sizeof(float)) + j*(nb01/sizeof(float)) + block_start;
+
+        // Use float4 vectorized loads for better memory bandwidth (8x float4 = 32 floats)
+        // DKQ is always multiple of 32 (static_assert at line 622), so no bounds check needed
+        const float4* Q_f4 = reinterpret_cast<const float4*>(Q_f + base_offset);
+        float4* Q_vals4 = reinterpret_cast<float4*>(Q_vals);
+
+        #pragma unroll
+        for (int i = 0; i < QK8_0/4; i++) {
+            float4 tmp = Q_f4[i];
+            tmp.x *= scale;
+            tmp.y *= scale;
+            tmp.z *= scale;
+            tmp.w *= scale;
+            Q_vals4[i] = tmp;
+        }
+
+        quantize_f32_q8_0_block(Q_vals, &Q_block);
+
+        Q_scales[col_block * ncols + jc] = Q_block.d;
+
+        int8_t * dst = Q_values + jc * DKQ + col_block * 32;
+        const int4* src_int4 = (const int4*)Q_block.qs;
+        int4* dst_int4 = (int4*)dst;
+
+        dst_int4[0] = src_int4[0];
+        dst_int4[1] = src_int4[1];
+    }
+
+    __syncthreads();
+}
+
+template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int nbatch_fa, int nbatch_K,
+    bool oob_check>
+static __device__ __forceinline__ void flash_attn_tile_q8_q8_iter_KQ(
+        int8_t * const Q_values,
+        half * const Q_scales,
+        const block_q8_0 * const __restrict__ K_q8,
+        int8_t * const K_values,
+        half * const K_scales,
+        const int stride_K_q8,
+        const int k_VKQ_0,
+        const int k_VKQ_sup,
+        const int k_KQ_0,
+        float * KQ_acc) {
+    constexpr int ncols = ncols1*ncols2;
+    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1;
+    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1;
+
+    constexpr int K_row_stride = nbatch_K + 16;
+
+    flash_attn_tile_q8_q8_load_tile_q8<warp_size, nwarps, nbatch_fa, nbatch_K, K_row_stride, oob_check>
+        (K_q8 + int64_t(k_VKQ_0)*stride_K_q8 + (k_KQ_0/32), K_values, K_scales, stride_K_q8, k_VKQ_sup);
+    __syncthreads();
+
+    static_assert(nbatch_K % 4 == 0, "nbatch_K must be multiple of 4 for sdot4");
+
+    constexpr int blocks_per_K_row = nbatch_K / 32;
+
+    #pragma unroll 4
+    for (int jc0 = 0; jc0 < cpw; ++jc0) {
+        const int jc = jc0 + (threadIdx.y / np)*cpw;
+
+        half q_scales_hoisted[blocks_per_K_row];
+        #pragma unroll
+        for (int block_id = 0; block_id < blocks_per_K_row; block_id++) {
+            q_scales_hoisted[block_id] = Q_scales[((k_KQ_0/32) + block_id) * ncols + jc];
+        }
+
+        #pragma unroll 4
+        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
+            const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;
+            const int idx = i_KQ_0/(np*warp_size)*cpw + jc0;
+
+            const int8_t* K_row_base = K_values + i_KQ * K_row_stride;
+
+            // Hoist K_scales outside block_id loop - reduces LDS accesses
+            half k_scales_hoisted[blocks_per_K_row];
+            #pragma unroll
+            for (int block_id = 0; block_id < blocks_per_K_row; block_id++) {
+                k_scales_hoisted[block_id] = K_scales[block_id * nbatch_fa + i_KQ];
+            }
+
+            // Increased unroll factor (2->4) reduces loop overhead for common head dims
+            #pragma unroll 4
+            for (int block_id = 0; block_id < blocks_per_K_row; block_id++) {
+                const int4* K_ptr4 = (const int4*)(K_row_base + block_id * 32);
+                const int4* Q_ptr4 = (const int4*)&Q_values[jc * DKQ + k_KQ_0 + block_id * 32];
+
+                int acc_int = 0;
+                const int4 K_lo = K_ptr4[0];
+                const int4 Q_lo = Q_ptr4[0];
+
+                acc_int = ggml_cuda_dp4a(K_lo.x, Q_lo.x, acc_int);
+                const int4 K_hi = K_ptr4[1];
+                acc_int = ggml_cuda_dp4a(K_lo.y, Q_lo.y, acc_int);
+                const int4 Q_hi = Q_ptr4[1];
+                acc_int = ggml_cuda_dp4a(K_lo.z, Q_lo.z, acc_int);
+                // Use hoisted K scale instead of LDS access
+                const half combined_scale_h = __hmul(k_scales_hoisted[block_id], q_scales_hoisted[block_id]);
+                acc_int = ggml_cuda_dp4a(K_lo.w, Q_lo.w, acc_int);
+
+                acc_int = ggml_cuda_dp4a(K_hi.x, Q_hi.x, acc_int);
+                acc_int = ggml_cuda_dp4a(K_hi.y, Q_hi.y, acc_int);
+                acc_int = ggml_cuda_dp4a(K_hi.z, Q_hi.z, acc_int);
+                acc_int = ggml_cuda_dp4a(K_hi.w, Q_hi.w, acc_int);
+
+                KQ_acc[idx] += __half2float(combined_scale_h) * (float)acc_int;
+            }
+        }
+    }
+
+    if (k_KQ_0 + nbatch_K < DKQ) {
+        __syncthreads();
+    }
+}
+
+template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int DV, int nbatch_fa, int nbatch_K,
+    bool use_logit_softcap, bool oob_check, typename T_KQ, typename T_acc>
+static __device__ __forceinline__ void flash_attn_tile_q8_q8_iter(
+        int8_t * const Q_values,
+        half * const Q_scales,
+        const block_q8_0 * const __restrict__ K_q8,
+        const half2 * const __restrict__ V_h2,
+        const half  * const __restrict__ mask,
+        const float logit_softcap,
+        const float slope,
+        T_KQ      * const KQ,
+        int8_t * const K_values,
+        half * const K_scales,
+        half2 * const V_tmp,
+        const int stride_K_q8,
+        const int stride_V2,
+        const int stride_mask,
+        float * const KQ_max,
+        float * const KQ_sum,
+        T_acc * const VKQ,
+        const int k_VKQ_0,
+        const int k_VKQ_max) {
+    constexpr int cpy_ne = ggml_cuda_get_max_cpy_bytes() / 4;
+
+    constexpr int ncols = ncols1*ncols2;
+    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1;
+    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1;
+
+    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1);
+
+    constexpr int KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
+    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
+    const int k_VKQ_sup = k_VKQ_max - k_VKQ_0;
+
+    float KQ_max_new[cpw];
+#pragma unroll
+    for (int jc0 = 0; jc0 < cpw; ++jc0) {
+        KQ_max_new[jc0] = KQ_max[jc0];
+    }
+
+    constexpr int num_i_KQ_iters = nbatch_fa/(np*warp_size);
+    float KQ_acc[num_i_KQ_iters * cpw] = {0.0f};
+
+    constexpr int nbatch_K_last = DKQ % nbatch_K;
+    constexpr int num_K_tiles = (DKQ - nbatch_K_last) / nbatch_K;
+
+    #pragma unroll
+    for (int tile = 0; tile < num_K_tiles; tile++) {
+        const int k_KQ_0 = tile * nbatch_K;
+        flash_attn_tile_q8_q8_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, oob_check>(
+            Q_values, Q_scales, K_q8, K_values, K_scales, stride_K_q8, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
+    }
+
+    if constexpr (nbatch_K_last > 0) {
+        constexpr int k_KQ_0 = DKQ - nbatch_K_last;
+        flash_attn_tile_q8_q8_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_last, oob_check>(
+            Q_values, Q_scales, K_q8, K_values, K_scales, stride_K_q8, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
+    }
+
+    if constexpr (num_i_KQ_iters == 1) {
+        const int i_KQ = (threadIdx.y % np)*warp_size + threadIdx.x;
+
+#pragma unroll
+        for (int jc0 = 0; jc0 < cpw; ++jc0) {
+            const int j = (jc0 + (threadIdx.y / np)*cpw)/ncols2;
+
+            if (use_logit_softcap) {
+                KQ_acc[jc0] = logit_softcap * fast_tanh_f32(KQ_acc[jc0]);
+            }
+
+            if (!oob_check || i_KQ < k_VKQ_sup) {
+                KQ_acc[jc0] += (ncols2 > 1 || mask) ?
+                    slope*__half2float(mask[j*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;
+
+                KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[jc0]);
+            }
+
+            KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
+        }
+    } else {
+#pragma unroll
+        for (int jc0 = 0; jc0 < cpw; ++jc0) {
+            const int j = (jc0 + (threadIdx.y / np)*cpw)/ncols2;
+
+#pragma unroll
+            for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
+                const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;
+
+                if (use_logit_softcap) {
+                    KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] = logit_softcap * fast_tanh_f32(KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0]);
+                }
+
+                if (!oob_check || i_KQ < k_VKQ_sup) {
+                    KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] += (ncols2 > 1 || mask) ?
+                        slope*__half2float(mask[j*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;
+
+                    KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0]);
+                }
+            }
+
+            KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
+        }
+    }
+
+    if constexpr (np == 1) {
+        __syncthreads();
+    } else {
+        static_assert(cpw == 1, "bad cpw");
+        __shared__ float KQ_max_new_shared[nwarps];
+        if (threadIdx.x == 0) {
+            KQ_max_new_shared[threadIdx.y] = KQ_max_new[0];
+        }
+        __syncthreads();
+        KQ_max_new[0] = KQ_max_new_shared[(threadIdx.y & ~(np-1)) + threadIdx.x % np];
+        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
+    }
+
+    if constexpr (num_i_KQ_iters == 1) {
+        const int i_KQ = (threadIdx.y % np)*warp_size + threadIdx.x;
+
+#pragma unroll
+        for (int jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
+            half tmp[1][KQ_cs];
+
+#pragma unroll
+            for (int jc1 = 0; jc1 < KQ_cs; ++jc1) {
+                const int jc = jc0 + jc1;
+
+                const float KQ_max_scale = fast_exp_f32(KQ_max[jc] - KQ_max_new[jc]);
+                KQ_max[jc] = KQ_max_new[jc];
+
+                const float val = !oob_check || i_KQ < k_VKQ_sup ?
+                    fast_exp_f32(KQ_acc[jc] - KQ_max[jc]) : 0.0f;
+                const float KQ_sum_add = val;
+                tmp[0][jc1] = val;
+
+                KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;
+
+                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
+#pragma unroll
+                for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
+                    VKQ[jc*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
+                }
+            }
+
+            ggml_cuda_memcpy_1<sizeof(tmp[0])>(
+                KQ + (jc0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs))*(nbatch_fa*KQ_cs) + i_KQ*KQ_cs,
+                tmp[0]);
+        }
+    } else {
+#pragma unroll
+        for (int jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
+            half tmp[num_i_KQ_iters][KQ_cs];
+
+#pragma unroll
+            for (int jc1 = 0; jc1 < KQ_cs; ++jc1) {
+                const int jc = jc0 + jc1;
+
+                const float KQ_max_scale = fast_exp_f32(KQ_max[jc] - KQ_max_new[jc]);
+                KQ_max[jc] = KQ_max_new[jc];
+
+                float KQ_sum_add = 0.0f;
+#pragma unroll
+                for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
+                    const float val = !oob_check || i0 + (threadIdx.y % np)*warp_size + threadIdx.x < k_VKQ_sup ?
+                        fast_exp_f32(KQ_acc[(i0/(np*warp_size))*cpw + jc] - KQ_max[jc]) : 0.0f;
+                    KQ_sum_add += val;
+                    tmp[i0/(np*warp_size)][jc1] = val;
+                }
+                KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;
+
+                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
+#pragma unroll
+                for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
+                    VKQ[jc*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
+                }
+            }
+
+#pragma unroll
+            for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
+                const int i = i0 + (threadIdx.y % np)*warp_size + threadIdx.x;
+
+                ggml_cuda_memcpy_1<sizeof(tmp[0])>(
+                    KQ + (jc0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs))*(nbatch_fa*KQ_cs) + i*KQ_cs,
+                    tmp[i0/(np*warp_size)]);
+            }
+        }
+    }
+
+    static_assert(DV <= DKQ, "bad DV");
+    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
+    constexpr int nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV;
+    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
+    static_assert(nbatch_V % np == 0, "bad nbatch_V");
+#pragma unroll
+    for (int k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
+        flash_attn_tile_q8_q8_load_tile<warp_size, nwarps, nbatch_V, DV, 0, oob_check>
+            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, V_tmp, stride_V2, k_VKQ_sup - k0);
+        __syncthreads();
+
+#pragma unroll
+        for (int k1 = 0; k1 < nbatch_V; k1 += np) {
+            half2 V_k[(DVp/2)/warp_size];
+            half2 KQ_k[cpw];
+
+            constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
+
+#pragma unroll
+            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
+                ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/warp_size], &V_tmp[(k1 + threadIdx.y % np)*(DV/2) + i0 + threadIdx.x*cpy_ne_D]);
+            }
+
+#pragma unroll
+            for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
+                const int jc_KQ = jc_VKQ_0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs);
+
+                half tmp[KQ_cs];
+                ggml_cuda_memcpy_1<KQ_cs*sizeof(half)>(
+                    &tmp, KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + threadIdx.y % np)*KQ_cs);
+#pragma unroll
+                for (int jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
+                    KQ_k[jc_VKQ_0+jc_VKQ_1] = __half2half2(tmp[jc_VKQ_1]);
+                }
+            }
+
+#pragma unroll
+            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
+                const half2 v_val = V_k[i0/warp_size];
+#pragma unroll
+                for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
+                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size] += v_val * KQ_k[jc_VKQ_0];
+                }
+            }
+        }
+
+        __syncthreads();
+    }
+}
+
+template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
+__launch_bounds__(ggml_cuda_fattn_tile_q8_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_tile_q8_get_occupancy(DKQ, DV, ncols1*ncols2))
+static __global__ void flash_attn_tile_q8(
+        const char * __restrict__ Q,
+        const char * __restrict__ K,
+        const char * __restrict__ V,
+        const char * __restrict__ mask,
+        const char * __restrict__ sinks,
+        const int  * __restrict__ KV_max,
+        float      * __restrict__ dst,
+        float2     * __restrict__ dst_meta,
+        const float scale,
+        const float max_bias,
+        const float m0,
+        const float m1,
+        const uint32_t n_head_log2,
+        const float logit_softcap,
+        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
+                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
+        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
+                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
+                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
+                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
+                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
+#ifdef FLASH_ATTN_AVAILABLE
+
+    if (use_logit_softcap && !(DV == 128 || DV == 256)) {
+        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
+            max_bias, m0, m1, n_head_log2, logit_softcap,
+            ne00, ne01, ne02, ne03,
+                  nb01, nb02, nb03,
+            ne10, ne11, ne12, ne13,
+                  nb11, nb12, nb13,
+                  nb21, nb22, nb23,
+                  ne31, ne32, ne33,
+                  nb31, nb32, nb33);
+        NO_DEVICE_CODE;
+        return;
+    }
+
+    static_assert(ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");
+    static_assert(DKQ % 32 == 0, "DKQ must be multiple of 32 for Q8_0 quantization");
+
+    constexpr int ncols     = ncols1*ncols2;
+    constexpr int warp_size = 32;
+    constexpr int nwarps    = ggml_cuda_fattn_tile_q8_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_size;
+    constexpr int nbatch_fa = ggml_cuda_fattn_tile_q8_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
+    constexpr int nbatch_K  = ggml_cuda_fattn_tile_q8_get_nbatch_K (DKQ, DV, ncols1*ncols2);
+
+    const int col_Q_0 = blockIdx.x * ncols1;
+
+    const int sequence = blockIdx.z / (ne02/ncols2);
+    const int head0 = blockIdx.z*ncols2 - sequence*ne02;
+    const int gqa_ratio = ne02 / ne12;
+    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02* head0              + nb01*col_Q_0);
+    const block_q8_0 * K_q8 = (const block_q8_0 *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
+    const half2 * V_h2 = (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio));
+
+    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33) + nb31*col_Q_0) : nullptr;
+
+    const int stride_K_q8 = nb11 / sizeof(block_q8_0);
+    const int stride_V2   = nb21 / sizeof(half2);
+    const int stride_mask = nb31 / sizeof(half);
+
+    float slope_tmp = 0.0f;
+    if (threadIdx.x == 0) {
+        slope_tmp = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;
+    }
+    const float slope = sgpr_broadcast_f32(slope_tmp);
+
+    constexpr int cpy_ne = ggml_cuda_get_max_cpy_bytes() / 4;
+
+    constexpr int cpw = ncols > nwarps ? ncols/nwarps : 1;
+    constexpr int np  = nwarps > ncols ? nwarps/ncols : 1;
+    static_assert(cpw == 1 || np == 1, "bad cpw / np");
+    static_assert(nbatch_fa % (np*warp_size) == 0, "nbatch_fa % (np*warp_size) != 0");
+
+    constexpr int DVp  = (DV  + 2*warp_size - 1) & ~(2*warp_size - 1);
+
+    __shared__ int8_t Q_values[ncols * DKQ];
+    __shared__ half   Q_scales[ncols * (DKQ/32)];
+
+    constexpr int K_row_padding = 16;
+    __shared__ int8_t K_values[nbatch_fa * (nbatch_K + K_row_padding)];
+    __shared__ half   K_scales[nbatch_fa * (nbatch_K/32)];
+
+    __shared__ half2 KV_tmp[nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV];
+
+    __shared__ half  KQ[ncols * nbatch_fa];
+    half2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};
+
+    float KQ_max[cpw];
+#pragma unroll
+    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
+        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
+    }
+    float KQ_sum[cpw] = {0.0f};
+
+    flash_attn_tile_q8_quantize_Q_to_shared<nwarps*warp_size, ncols, ncols2, DKQ>(
+        Q_f, Q_values, Q_scales, col_Q_0, int(ne01.z), nb01, nb02, scale);
+
+    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
+    if (ncols2 == 1) {
+        int k_VKQ_0 = blockIdx.y*nbatch_fa;
+        while (k_VKQ_0 < k_VKQ_max - nbatch_fa) {
+            constexpr bool oob_check = false;
+            flash_attn_tile_q8_q8_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
+                (Q_values, Q_scales, K_q8, V_h2, maskh, logit_softcap, slope, KQ, K_values, K_scales, KV_tmp,
+                stride_K_q8, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max);
+            k_VKQ_0 += gridDim.y*nbatch_fa;
+        }
+        if (k_VKQ_0 < k_VKQ_max) {
+            constexpr bool oob_check = true;
+            flash_attn_tile_q8_q8_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
+                (Q_values, Q_scales, K_q8, V_h2, maskh, logit_softcap, slope, KQ, K_values, K_scales, KV_tmp,
+                stride_K_q8, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max);
+        }
+    } else {
+        for (int k_VKQ_0 = blockIdx.y*nbatch_fa; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nbatch_fa) {
+            constexpr bool oob_check = false;
+            flash_attn_tile_q8_q8_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
+                (Q_values, Q_scales, K_q8, V_h2, maskh, logit_softcap, slope, KQ, K_values, K_scales, KV_tmp,
+                stride_K_q8, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max);
+        }
+    }
+
+#pragma unroll
+    for (int jc0 = 0; jc0 < cpw; ++jc0) {
+        KQ_sum[jc0] = warp_reduce_sum<warp_size>(KQ_sum[jc0]);
+    }
+
+    if constexpr (np > 1) {
+        static_assert(cpw == 1, "bad cpw");
+        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");
+
+        half2 * VKQ_combine    = (half2 *) KV_tmp;
+        float * KQ_sum_combine = (float *) Q_values;
+
+        if (threadIdx.y % np != 0) {
+            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
+#pragma unroll
+            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
+                ggml_cuda_memcpy_1<cpy_ne_D*4>(&VKQ_combine[threadIdx.y*(DVp/2) + i0 + threadIdx.x*cpy_ne_D], &VKQ[i0/warp_size]);
+            }
+
+            if (threadIdx.x == 0) {
+                KQ_sum_combine[threadIdx.y] = KQ_sum[0];
+            }
+
+            return;
+        }
+
+        __syncthreads();
+
+#pragma unroll
+        for (int ip = 1; ip < np; ++ip) {
+            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
+#pragma unroll
+            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
+                half2 tmp[cpy_ne_D];
+                ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(threadIdx.y + ip)*(DVp/2) + i0 + threadIdx.x*cpy_ne_D]);
+#pragma unroll
+                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
+                    VKQ[i0/warp_size + i1] += tmp[i1];
+                }
+            }
+
+            KQ_sum[0] += KQ_sum_combine[threadIdx.y + ip];
+        }
+    }
+
+    if (sinks && blockIdx.y == 0) {
+#pragma unroll
+        for (int jc0 = 0; jc0 < cpw; ++jc0) {
+            const int jc = jc0 + (threadIdx.y/np)*cpw;
+            const float sink = ((const float *) sinks)[head0 + jc % ncols2];
+
+            float KQ_max_new_j = fmaxf(KQ_max[jc0], sink);
+            const float KQ_max_scale = fast_exp_f32(KQ_max[jc0] - KQ_max_new_j);
+            KQ_max[jc0] = KQ_max_new_j;
+
+            const float val = fast_exp_f32(sink - KQ_max[jc0]);
+            KQ_sum[jc0] = KQ_sum[jc0]*KQ_max_scale + val;
+
+            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
+#pragma unroll
+            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
+                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
+            }
+        }
+    }
+
+#pragma unroll
+    for (int jc0 = 0; jc0 < cpw; ++jc0) {
+        const int jc = jc0 + (threadIdx.y/np)*cpw;
+
+        const int j = jc / ncols2;
+        const int c = jc % ncols2;
+
+        if (ncols1 > 1 && col_Q_0 + j >= int(ne01.z)) {
+            return;
+        }
+
+        const float scale = gridDim.y == 1 ? 1.0f/KQ_sum[jc0] : 1.0f;
+
+        const int j_dst_unrolled = ((sequence*int(ne01.z) + col_Q_0 + j)*ne02 + head0 + c)*gridDim.y + blockIdx.y;
+
+        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
+#pragma unroll
+        for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
+            float2 tmp[cpy_ne_D];
+#pragma unroll
+            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
+                tmp[i1] = __half22float2(VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size + i1]);
+                tmp[i1].x *= scale;
+                tmp[i1].y *= scale;
+            }
+            if (i0 + warp_size*cpy_ne_D <= DV/2 || i0 + threadIdx.x*cpy_ne_D < DV/2) {
+                ggml_cuda_memcpy_1<sizeof(tmp)>(&dst[j_dst_unrolled*DV + 2*i0 + threadIdx.x*(2*cpy_ne_D)], tmp);
+            }
+        }
+
+        if (gridDim.y != 1 && threadIdx.x == 0) {
+            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_sum[jc0]);
+        }
+    }
+#else
+    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
+        max_bias, m0, m1, n_head_log2, logit_softcap,
+        ne00, ne01, ne02, ne03,
+              nb01, nb02, nb03,
+        ne10, ne11, ne12, ne13,
+              nb11, nb12, nb13,
+              nb21, nb22, nb23,
+              ne31, ne32, ne33,
+              nb31, nb32, nb33);
+    NO_DEVICE_CODE;
+#endif
+}
+
+template <int DKQ, int DV, int ncols2, bool use_logit_softcap>
+static void launch_fattn_tile_q8_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
+    const ggml_tensor * Q = dst->src[0];
+
+    const int id        = ggml_cuda_get_device();
+    const int cc        = ggml_cuda_info().devices[id].cc;
+    const int warp_size = 32;
+
+    constexpr size_t nbytes_shared = 0;
+
+    if constexpr (DV <= 128) {
+        if (Q->ne[1] > 32/ncols2) {
+            constexpr int cols_per_block = 64;
+            const int nwarps    = ggml_cuda_fattn_tile_q8_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
+            const int nbatch_fa = ggml_cuda_fattn_tile_q8_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
+            fattn_kernel_t fattn_kernel = flash_attn_tile_q8<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap>;
+            launch_fattn<DV, cols_per_block/ncols2, ncols2>
+                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, false, true, false, warp_size);
+            return;
+        }
+    }
+
+    {
+        if (Q->ne[1] > 16/ncols2) {
+            constexpr int cols_per_block = 32;
+            const int nwarps    = ggml_cuda_fattn_tile_q8_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
+            const int nbatch_fa = ggml_cuda_fattn_tile_q8_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
+            fattn_kernel_t fattn_kernel = flash_attn_tile_q8<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap>;
+            launch_fattn<DV, cols_per_block/ncols2, ncols2>
+                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, false, true, false, warp_size);
+            return;
+        }
+    }
+
+    if (Q->ne[1] > 8/ncols2) {
+        constexpr int cols_per_block = 16;
+        const int nwarps    = ggml_cuda_fattn_tile_q8_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
+        const int nbatch_fa = ggml_cuda_fattn_tile_q8_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
+        fattn_kernel_t fattn_kernel = flash_attn_tile_q8<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap>;
+        launch_fattn<DV, cols_per_block/ncols2, ncols2>
+            (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, false, true, false, warp_size);
+        return;
+    }
+
+    if constexpr (ncols2 <= 8) {
+        if (Q->ne[1] > 4/ncols2) {
+            constexpr int cols_per_block = 8;
+            const int nwarps    = ggml_cuda_fattn_tile_q8_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
+            const int nbatch_fa = ggml_cuda_fattn_tile_q8_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
+            fattn_kernel_t fattn_kernel = flash_attn_tile_q8<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap>;
+            launch_fattn<DV, cols_per_block/ncols2, ncols2>
+                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, false, true, false, warp_size);
+            return;
+        }
+    }
+
+    if constexpr (ncols2 <= 4) {
+        if (Q->ne[1] > 2/ncols2) {
+            constexpr int cols_per_block = 4;
+            const int nwarps    = ggml_cuda_fattn_tile_q8_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
+            const int nbatch_fa = ggml_cuda_fattn_tile_q8_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
+            fattn_kernel_t fattn_kernel = flash_attn_tile_q8<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap>;
+            launch_fattn<DV, cols_per_block/ncols2, ncols2>
+                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, false, true, false, warp_size);
+            return;
+        }
+    }
+
+    if constexpr (ncols2 <= 2) {
+        constexpr int cols_per_block = 2;
+        const int nwarps    = ggml_cuda_fattn_tile_q8_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
+        const int nbatch_fa = ggml_cuda_fattn_tile_q8_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
+        fattn_kernel_t fattn_kernel = flash_attn_tile_q8<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap>;
+        launch_fattn<DV, cols_per_block/ncols2, ncols2>
+            (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, false, true, false, warp_size);
+        return;
+    }
+
+    GGML_ABORT("fatal error");
+}
+
+template <int DKQ, int DV, bool use_logit_softcap>
+static void launch_fattn_tile_q8_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
+    const ggml_tensor * KQV  = dst;
+    const ggml_tensor * Q    = dst->src[0];
+    const ggml_tensor * K    = dst->src[1];
+    const ggml_tensor * mask = dst->src[3];
+
+    float max_bias = 0.0f;
+    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
+
+    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
+    const int gqa_ratio = Q->ne[2] / K->ne[2];
+
+    const bool nvidia = GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
+    const int gqa_limit = nvidia && gqa_ratio <= 4 ? 16 : INT_MAX;
+    const bool use_gqa_opt = mask && max_bias == 0.0f && Q->ne[1] <= gqa_limit && K->ne[1] % FATTN_KQ_STRIDE == 0;
+
+    if constexpr (DV == 512) {
+        if (use_gqa_opt && gqa_ratio % 16 == 0) {
+            launch_fattn_tile_q8_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst);
+            return;
+        }
+    }
+
+    if constexpr (DV <= 256) {
+        if (use_gqa_opt && gqa_ratio % 8 == 0) {
+            launch_fattn_tile_q8_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
+            return;
+        }
+
+        if (use_gqa_opt && gqa_ratio % 4 == 0) {
+            launch_fattn_tile_q8_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
+            return;
+        }
+
+        if (use_gqa_opt && gqa_ratio % 2 == 0) {
+            launch_fattn_tile_q8_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
+            return;
+        }
+
+        launch_fattn_tile_q8_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
+        return;
+    }
+    GGML_ABORT("fatal error");
+}
+
+template <int DKQ, int DV>
+void ggml_cuda_flash_attn_ext_tile_q8_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
+    const ggml_tensor * KQV = dst;
+
+    float logit_softcap;
+    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
+
+    if (logit_softcap == 0.0f) {
+        constexpr bool use_logit_softcap = false;
+        launch_fattn_tile_q8_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
+    } else {
+        constexpr bool use_logit_softcap = true;
+        launch_fattn_tile_q8_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
+    }
+}
+
+void ggml_cuda_flash_attn_ext_tile_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
+
+#define DECL_FATTN_TILE_CASE(DKQ, DV)                             \
+    template void ggml_cuda_flash_attn_ext_tile_q8_case              \
+    <DKQ, DV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \
+
+extern DECL_FATTN_TILE_CASE( 40,  40);
+extern DECL_FATTN_TILE_CASE( 64,  64);
+extern DECL_FATTN_TILE_CASE( 80,  80);
+extern DECL_FATTN_TILE_CASE( 96,  96);
+extern DECL_FATTN_TILE_CASE(112, 112);
+extern DECL_FATTN_TILE_CASE(128, 128);
+extern DECL_FATTN_TILE_CASE(256, 256);
diff --git a/ggml/src/ggml-cuda/gfx906/gfx906-mmq-prefetch.cuh b/ggml/src/ggml-cuda/gfx906/gfx906-mmq-prefetch.cuh
new file mode 100644
index 000000000..013f5d9b5
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/gfx906-mmq-prefetch.cuh
@@ -0,0 +1,149 @@
+#pragma once
+
+// Y-tile prefetch for MMQ: issues global_load_dword for next iteration
+// Hides memory latency by overlapping loads with compute
+
+#include "gfx906-config.h"
+
+#if defined(GGML_USE_HIP) && defined(__gfx906__)
+
+template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
+static __device__ __forceinline__ int gfx906_prefetch_y_tile_v4(
+    const int * __restrict__ y,
+    const int ncols_y,
+    const int kb0,
+    const int kb0_stop,
+    const int qk,
+    const int blocks_per_iter) {
+
+    if (threadIdx.y != 0) {
+        return 0;
+    }
+
+    const int kb0_next = kb0 + blocks_per_iter;
+
+    if (kb0_next >= kb0_stop) {
+        return 0;
+    }
+
+    constexpr int block_q8_1_mmq_bytes = 144;
+    constexpr int QK8_1_val = 32;
+    const int stride_factor = qk * block_q8_1_mmq_bytes / (4 * QK8_1_val * sizeof(int));
+
+    const int * by_next = y + ncols_y * (kb0_next * stride_factor);
+
+    const int lane_id = threadIdx.x;
+    if (lane_id >= 2) {
+        return 0;
+    }
+
+    const int prefetch_offset = lane_id * 256;
+    const int * prefetch_addr = by_next + prefetch_offset;
+
+    int prefetch_data;
+    asm volatile(
+        "global_load_dword %0, %1, off\n"
+        : "=v"(prefetch_data)
+        : "v"(prefetch_addr)
+        : "memory"
+    );
+    return prefetch_data;
+}
+
+static __device__ __forceinline__ void gfx906_prefetch_consume(int prefetch_data) {
+    asm volatile(
+        "v_mov_b32 %0, %0\n"
+        : "+v"(prefetch_data)
+    );
+}
+
+template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
+static __device__ __forceinline__ void gfx906_prefetch_y_tile_v2(
+    const int * __restrict__ y,
+    const int ncols_y,
+    const int kb0,
+    const int kb0_stop,
+    const int qk,
+    const int blocks_per_iter) {
+
+    const int kb0_next = kb0 + blocks_per_iter;
+
+    if (kb0_next >= kb0_stop) {
+        return;
+    }
+
+    const int tid = threadIdx.y * warp_size + threadIdx.x;
+
+    constexpr int total_elements = mmq_x * mmq_tile_y_k;
+    if (tid >= total_elements) {
+        return;
+    }
+
+    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
+
+    constexpr int block_q8_1_mmq_bytes = 144;
+    constexpr int QK8_1_val = 32;
+    const int stride_factor = qk * block_q8_1_mmq_bytes / (4 * QK8_1_val * sizeof(int));
+
+    const int * by_next = y + ncols_y * (kb0_next * stride_factor);
+    const int * prefetch_addr = by_next + tid;
+
+    int dummy;
+    asm volatile(
+        "global_load_dword %0, %1, off\n"
+        : "=v"(dummy)
+        : "v"(prefetch_addr)
+        : "memory"
+    );
+}
+
+template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
+static __device__ __forceinline__ void gfx906_prefetch_y_tile_v1(
+    const int * __restrict__ y,
+    const int ncols_y,
+    const int kb0,
+    const int kb0_stop,
+    const int qk,
+    const int blocks_per_iter) {
+
+    const int kb0_next = kb0 + blocks_per_iter;
+
+    if (kb0_next >= kb0_stop) {
+        return;
+    }
+
+    const int tid = threadIdx.y * warp_size + threadIdx.x;
+    constexpr int total_elements = mmq_x * mmq_tile_y_k;
+
+    if (tid >= total_elements) {
+        return;
+    }
+
+    constexpr int block_q8_1_mmq_bytes = 144;
+    constexpr int QK8_1_val = 32;
+    const int stride_factor = qk * block_q8_1_mmq_bytes / (4 * QK8_1_val * sizeof(int));
+
+    const int * by_next = y + ncols_y * (kb0_next * stride_factor);
+    const int * prefetch_addr = by_next + tid;
+
+    int dummy;
+    asm volatile(
+        "global_load_dword %0, %1, off\n"
+        : "=v"(dummy)
+        : "v"(prefetch_addr)
+        : "memory"
+    );
+}
+
+template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
+static __device__ __forceinline__ void gfx906_prefetch_y_tile_noop(
+    const int * __restrict__ y,
+    const int ncols_y,
+    const int kb0,
+    const int kb0_stop,
+    const int qk,
+    const int blocks_per_iter) {
+    (void)y; (void)ncols_y; (void)kb0; (void)kb0_stop; (void)qk; (void)blocks_per_iter;
+}
+
+#endif // defined(GGML_USE_HIP) && defined(__gfx906__)
diff --git a/ggml/src/ggml-cuda/gfx906/gfx906-mmq.cuh b/ggml/src/ggml-cuda/gfx906/gfx906-mmq.cuh
new file mode 100644
index 000000000..2d4038d60
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/gfx906-mmq.cuh
@@ -0,0 +1,93 @@
+#pragma once
+
+// MMQ vectorized loads: 2x int4 (128-bit) instead of 8x scalar loads
+// Q8_0 software pipelining: separate load/store phases for better MLP
+
+#include "gfx906-config.h"
+#include "gfx906-vecdotq.cuh"
+
+#if defined(GGML_USE_HIP)
+
+static __device__ __forceinline__ void gfx906_load_q4_0_quants_vectorized(
+    const int * __restrict__ y_qs,
+    const int base_addr,
+    const int qi,
+    int * __restrict__ u) {
+
+    const int4 vec0 = *((const int4 *) &y_qs[base_addr]);
+    const int4 vec1 = *((const int4 *) &y_qs[base_addr + qi]);
+
+    u[0] = vec0.x; u[2] = vec0.y; u[4] = vec0.z; u[6] = vec0.w;
+    u[1] = vec1.x; u[3] = vec1.y; u[5] = vec1.z; u[7] = vec1.w;
+}
+
+static __device__ __forceinline__ void gfx906_load_q4_1_quants_vectorized(
+    const int * __restrict__ y_qs,
+    const int base_addr,
+    const int qi,
+    int * __restrict__ u) {
+
+    const int4 vec0 = *((const int4 *) &y_qs[base_addr]);
+    const int4 vec1 = *((const int4 *) &y_qs[base_addr + qi]);
+
+    u[0] = vec0.x; u[2] = vec0.y; u[4] = vec0.z; u[6] = vec0.w;
+    u[1] = vec1.x; u[3] = vec1.y; u[5] = vec1.z; u[7] = vec1.w;
+}
+
+template<int VDR>
+static __device__ __forceinline__ void gfx906_load_quants_vectorized(
+    const int * __restrict__ y_qs,
+    const int base_addr,
+    const int qi,
+    int * __restrict__ u) {
+
+    static_assert(VDR == 4, "Only VDR=4 supported for vectorized loads");
+
+    const int4 vec0 = *((const int4 *) &y_qs[base_addr]);
+    const int4 vec1 = *((const int4 *) &y_qs[base_addr + qi]);
+
+    u[0] = vec0.x; u[2] = vec0.y; u[4] = vec0.z; u[6] = vec0.w;
+    u[1] = vec1.x; u[3] = vec1.y; u[5] = vec1.z; u[7] = vec1.w;
+}
+
+#if defined(__gfx906__)
+
+#define GFX906_LOAD_TILES_Q8_0_ASYNC(cache_size, nrows, nwarps, threads_per_row, need_check, \
+    x, kbx0, stride, i_max, txi, kbx, kqsx, qs0_cache, qs1_cache, i_slot_cache) \
+    do { \
+        _Pragma("unroll") \
+        for (int iter = 0; iter < cache_size; iter++) { \
+            const int i0 = iter * nrows * nwarps; \
+            const int i_slot = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row); \
+            const int i_read = need_check ? min(i_slot, i_max) : i_slot; \
+            const bool oob = need_check && (i_slot > i_max); \
+            const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i_read*stride + kbx; \
+            qs0_cache[iter] = oob ? 0 : gfx906_get_int_b2_fast(bxi[0].qs, kqsx); \
+            qs1_cache[iter] = oob ? 0 : gfx906_get_int_b2_fast(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx); \
+            i_slot_cache[iter] = i_slot; \
+        } \
+    } while(0)
+
+#define GFX906_STORE_TILES_Q8_0_LDS_MMA(cache_size, x_qs, qs0_cache, qs1_cache, i_slot_cache, txi) \
+    do { \
+        _Pragma("unroll") \
+        for (int iter = 0; iter < cache_size; iter++) { \
+            const int i_slot = i_slot_cache[iter]; \
+            x_qs[i_slot*MMQ_MMA_TILE_X_K_Q8_0 + 0             + txi] = qs0_cache[iter]; \
+            x_qs[i_slot*MMQ_MMA_TILE_X_K_Q8_0 + MMQ_TILE_NE_K + txi] = qs1_cache[iter]; \
+        } \
+    } while(0)
+
+#define GFX906_STORE_TILES_Q8_0_LDS_LEGACY(cache_size, x_qs, qs0_cache, qs1_cache, i_slot_cache, txi) \
+    do { \
+        _Pragma("unroll") \
+        for (int iter = 0; iter < cache_size; iter++) { \
+            const int i_slot = i_slot_cache[iter]; \
+            x_qs[i_slot*(2*MMQ_TILE_NE_K + 1) + 0             + txi] = qs0_cache[iter]; \
+            x_qs[i_slot*(2*MMQ_TILE_NE_K + 1) + MMQ_TILE_NE_K + txi] = qs1_cache[iter]; \
+        } \
+    } while(0)
+
+#endif // defined(__gfx906__)
+
+#endif // GGML_USE_HIP
diff --git a/ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q4_0.cuh b/ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q4_0.cuh
new file mode 100644
index 000000000..0d2f5b158
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q4_0.cuh
@@ -0,0 +1,119 @@
+#pragma once
+
+// GFX906 Warp-Cooperative Q4_1 GEMV Kernel
+// Uses half-warp (32 threads) per row for better memory coalescing
+// Achieves better bandwidth improvement over sequential per-thread approach, which results in faster performance for small matrixes (ncols less than 1024)
+// This kernel is only included from mmvq.cu where all dependencies are available
+
+#if defined(GGML_USE_HIP)
+
+__launch_bounds__(64, 1)
+static __global__ void gfx906_mul_mat_vec_q4_0_warp_coop(
+        const void * __restrict__ vx, const void * __restrict__ vy,
+        const int32_t * __restrict__ ids,
+        float * __restrict__ dst,
+        const uint32_t ncols_x, const uint3 nchannels_y,
+        const uint32_t stride_row_x,
+        const uint32_t stride_col_dst, const uint3 channel_ratio,
+        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
+        const uint32_t stride_channel_dst, const uint3 sample_ratio,
+        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
+        const uint32_t stride_sample_dst, const uint32_t nrows_x) {
+
+    constexpr int qk_q4_0 = 32;
+
+    const int lane_id = threadIdx.x;
+    const int half_lane = lane_id % 32;
+    const int row_offset = lane_id / 32;
+
+    const int row = blockIdx.x * 2 + row_offset;
+
+    if (row >= (int)nrows_x) return;
+
+    const uint32_t channel_dst = blockIdx.y;
+    const uint32_t channel_x   = ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
+    const uint32_t channel_y   = ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
+    const uint32_t sample_dst  = blockIdx.z;
+    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
+    const uint32_t sample_y    = sample_dst;
+
+    const int blocks_per_row = ncols_x / qk_q4_0;
+    const int kbx_offset = sample_x * stride_sample_x + channel_x * stride_channel_x + row * stride_row_x;
+
+    const block_q4_0 * x = (const block_q4_0 *)vx + kbx_offset;
+    const block_q8_1 * y = (const block_q8_1 *)vy + sample_y * stride_sample_y + channel_y * stride_channel_y;
+
+    float sumf = 0.0f;
+
+    for (int ib = half_lane; ib < blocks_per_row; ib += 32) {
+        const block_q4_0 * bq4 = x + ib;
+        const block_q8_1 * bq8 = y + ib;
+
+        // Load 16 bytes of Q4_0 quantized values (32 nibbles)
+        int v0, v1, v2, v3;
+        memcpy(&v0, bq4->qs +  0, 4);
+        memcpy(&v1, bq4->qs +  4, 4);
+        memcpy(&v2, bq4->qs +  8, 4);
+        memcpy(&v3, bq4->qs + 12, 4);
+
+        // Load 32 bytes of Q8_1 quantized values
+        const int * q8 = (const int *)bq8->qs;
+        const int u0 = q8[0];
+        const int u1 = q8[1];
+        const int u2 = q8[2];
+        const int u3 = q8[3];
+        const int u4 = q8[4];
+        const int u5 = q8[5];
+        const int u6 = q8[6];
+        const int u7 = q8[7];
+
+        // Compute dot product (8 dp4a for full 32 values)
+        int sumi = 0;
+        sumi = ggml_cuda_dp4a((v0 >> 0) & 0x0F0F0F0F, u0, sumi);
+        sumi = ggml_cuda_dp4a((v0 >> 4) & 0x0F0F0F0F, u4, sumi);
+        sumi = ggml_cuda_dp4a((v1 >> 0) & 0x0F0F0F0F, u1, sumi);
+        sumi = ggml_cuda_dp4a((v1 >> 4) & 0x0F0F0F0F, u5, sumi);
+        sumi = ggml_cuda_dp4a((v2 >> 0) & 0x0F0F0F0F, u2, sumi);
+        sumi = ggml_cuda_dp4a((v2 >> 4) & 0x0F0F0F0F, u6, sumi);
+        sumi = ggml_cuda_dp4a((v3 >> 0) & 0x0F0F0F0F, u3, sumi);
+        sumi = ggml_cuda_dp4a((v3 >> 4) & 0x0F0F0F0F, u7, sumi);
+
+        // Q4_0 formula: d4 * (sumi * d8 - 8 * s8)
+        // where s8 is the sum of Q8 values (stored in bq8->ds.y)
+        const float d4 = bq4->d;
+        const float2 ds8 = __half22float2(bq8->ds);
+        sumf += d4 * (sumi * ds8.x - 8.0f * ds8.y);
+    }
+
+    // Half-warp reduction using fused DPP instructions
+    sumf = warp_reduce_sum<32>(sumf);
+
+    if (half_lane == 0) {
+        dst[sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row] = sumf;
+    }
+}
+
+static void gfx906_launch_mul_mat_vec_q4_0_warp_coop(
+        const void * vx, const void * vy, const int32_t * ids,
+        float * dst,
+        const uint32_t ncols_x, const uint3 nchannels_y,
+        const uint32_t stride_row_x,
+        const uint32_t stride_col_dst, const uint3 channel_ratio,
+        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
+        const uint32_t stride_channel_dst, const uint3 sample_ratio,
+        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
+        const uint32_t stride_sample_dst, const uint32_t nrows_x,
+        const uint32_t nchannels_dst, const uint32_t nsamples_dst,
+        cudaStream_t stream) {
+
+    const dim3 block_dims(64, 1, 1);
+    const dim3 block_nums((nrows_x + 1) / 2, nchannels_dst, nsamples_dst);
+
+    gfx906_mul_mat_vec_q4_0_warp_coop<<<block_nums, block_dims, 0, stream>>>(
+        vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x,
+        stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y,
+        stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
+        stride_sample_dst, nrows_x);
+}
+
+#endif // GGML_USE_HIP
diff --git a/ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q4_1.cuh b/ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q4_1.cuh
new file mode 100644
index 000000000..72752554c
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q4_1.cuh
@@ -0,0 +1,122 @@
+#pragma once
+
+// GFX906 Warp-Cooperative Q4_1 GEMV Kernel
+// Uses half-warp (32 threads) per row for better memory coalescing
+// Achieves better bandwidth improvement over sequential per-thread approach, which results in faster performance for small matrixes (ncols less than 1024)
+// This kernel is only included from mmvq.cu where all dependencies are available
+
+#if defined(GGML_USE_HIP)
+
+__launch_bounds__(64, 1)
+static __global__ void gfx906_mul_mat_vec_q4_1_warp_coop(
+        const void * __restrict__ vx, const void * __restrict__ vy,
+        const int32_t * __restrict__ ids,
+        float * __restrict__ dst,
+        const uint32_t ncols_x, const uint3 nchannels_y,
+        const uint32_t stride_row_x,
+        const uint32_t stride_col_dst, const uint3 channel_ratio,
+        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
+        const uint32_t stride_channel_dst, const uint3 sample_ratio,
+        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
+        const uint32_t stride_sample_dst, const uint32_t nrows_x) {
+
+    constexpr int qk_q4_1 = 32;
+
+    const int lane_id = threadIdx.x;
+    const int half_lane = lane_id % 32;      
+    const int row_offset = lane_id / 32;     
+    const int row = blockIdx.x * 2 + row_offset;
+
+    if (row >= (int)nrows_x) return;
+
+    const uint32_t channel_dst = blockIdx.y;
+    const uint32_t channel_x   = ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
+    const uint32_t channel_y   = ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
+    const uint32_t sample_dst  = blockIdx.z;
+    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
+    const uint32_t sample_y    = sample_dst;
+
+    const int blocks_per_row = ncols_x / qk_q4_1;
+    const int kbx_offset = sample_x * stride_sample_x + channel_x * stride_channel_x + row * stride_row_x;
+
+    const block_q4_1 * x = (const block_q4_1 *)vx + kbx_offset;
+    const block_q8_1 * y = (const block_q8_1 *)vy + sample_y * stride_sample_y + channel_y * stride_channel_y;
+
+    float sumf = 0.0f;
+
+    for (int ib = half_lane; ib < blocks_per_row; ib += 32) {
+        const block_q4_1 * bq4 = x + ib;
+        const block_q8_1 * bq8 = y + ib;  
+
+        // Load ALL 16 bytes of Q4_1 quantized values (32 nibbles = 32 values)
+        int v0, v1, v2, v3;
+        memcpy(&v0, bq4->qs +  0, 4);  // bytes 0-3:   values 0-3 (low), 16-19 (high)
+        memcpy(&v1, bq4->qs +  4, 4);  // bytes 4-7:   values 4-7 (low), 20-23 (high)
+        memcpy(&v2, bq4->qs +  8, 4);  // bytes 8-11:  values 8-11 (low), 24-27 (high)
+        memcpy(&v3, bq4->qs + 12, 4);  // bytes 12-15: values 12-15 (low), 28-31 (high)
+
+        // Load ALL 32 bytes of Q8_1 quantized values (32 int8 values)
+        const int * q8 = (const int *)bq8->qs;
+        const int u0 = q8[0];  // Q8 values 0-3
+        const int u1 = q8[1];  // Q8 values 4-7
+        const int u2 = q8[2];  // Q8 values 8-11
+        const int u3 = q8[3];  // Q8 values 12-15
+        const int u4 = q8[4];  // Q8 values 16-19
+        const int u5 = q8[5];  // Q8 values 20-23
+        const int u6 = q8[6];  // Q8 values 24-27
+        const int u7 = q8[7];  // Q8 values 28-31
+
+        // Compute dot product with nibble extraction (8 dp4a for full 32 values)
+        int sumi = 0;
+        sumi = ggml_cuda_dp4a((v0 >> 0) & 0x0F0F0F0F, u0, sumi);  // Q4 0-3 * Q8 0-3
+        sumi = ggml_cuda_dp4a((v0 >> 4) & 0x0F0F0F0F, u4, sumi);  // Q4 16-19 * Q8 16-19
+        sumi = ggml_cuda_dp4a((v1 >> 0) & 0x0F0F0F0F, u1, sumi);  // Q4 4-7 * Q8 4-7
+        sumi = ggml_cuda_dp4a((v1 >> 4) & 0x0F0F0F0F, u5, sumi);  // Q4 20-23 * Q8 20-23
+        sumi = ggml_cuda_dp4a((v2 >> 0) & 0x0F0F0F0F, u2, sumi);  // Q4 8-11 * Q8 8-11
+        sumi = ggml_cuda_dp4a((v2 >> 4) & 0x0F0F0F0F, u6, sumi);  // Q4 24-27 * Q8 24-27
+        sumi = ggml_cuda_dp4a((v3 >> 0) & 0x0F0F0F0F, u3, sumi);  // Q4 12-15 * Q8 12-15
+        sumi = ggml_cuda_dp4a((v3 >> 4) & 0x0F0F0F0F, u7, sumi);  // Q4 28-31 * Q8 28-31
+
+        // Load and apply scale/bias: Q4_1 has (d, m), Q8_1 has (d, s)
+        // Full block formula: result = sumi * d4 * d8 + m4 * s8
+        const float2 dm4 = __half22float2(bq4->dm);
+        const float2 ds8 = __half22float2(bq8->ds);
+        sumf += sumi * dm4.x * ds8.x + dm4.y * ds8.y;
+    }
+
+    // Half-warp reduction using fused DPP instructions
+    sumf = warp_reduce_sum<32>(sumf);
+
+    // First thread of each half-warp writes result
+    if (half_lane == 0) {
+        dst[sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row] = sumf;
+    }
+}
+
+// Host-side dispatch function for GFX906 Q4_1 warp-cooperative kernel
+static void gfx906_launch_mul_mat_vec_q4_1_warp_coop(
+        const void * vx, const void * vy, const int32_t * ids,
+        float * dst,
+        const uint32_t ncols_x, const uint3 nchannels_y,
+        const uint32_t stride_row_x,
+        const uint32_t stride_col_dst, const uint3 channel_ratio,
+        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
+        const uint32_t stride_channel_dst, const uint3 sample_ratio,
+        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
+        const uint32_t stride_sample_dst, const uint32_t nrows_x,
+        const uint32_t nchannels_dst, const uint32_t nsamples_dst,
+        cudaStream_t stream) {
+
+    // 2 rows per block, 64 threads per block (1 warp processing 2 rows)
+    const dim3 block_dims(64, 1, 1);
+    const dim3 block_nums((nrows_x + 1) / 2, nchannels_dst, nsamples_dst);
+
+    gfx906_mul_mat_vec_q4_1_warp_coop<<<block_nums, block_dims, 0, stream>>>(
+        vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x,
+        stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y,
+        stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
+        stride_sample_dst, nrows_x);
+}
+
+#endif // GGML_USE_HIP
+ 
\ No newline at end of file
diff --git a/ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q8_0.cuh b/ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q8_0.cuh
new file mode 100644
index 000000000..08bf4d92a
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q8_0.cuh
@@ -0,0 +1,120 @@
+#pragma once
+
+// GFX906 Warp-Cooperative Q8_0 GEMV Kernel
+// Simpler than Q4 variants - no nibble extraction needed
+
+#if defined(GGML_USE_HIP)
+
+__launch_bounds__(64, 1)
+static __global__ void gfx906_mul_mat_vec_q8_0_warp_coop(
+        const void * __restrict__ vx, const void * __restrict__ vy,
+        const int32_t * __restrict__ ids,
+        float * __restrict__ dst,
+        const uint32_t ncols_x, const uint3 nchannels_y,
+        const uint32_t stride_row_x,
+        const uint32_t stride_col_dst, const uint3 channel_ratio,
+        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
+        const uint32_t stride_channel_dst, const uint3 sample_ratio,
+        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
+        const uint32_t stride_sample_dst, const uint32_t nrows_x) {
+
+    constexpr int qk_q8_0 = 32;
+
+    const int lane_id = threadIdx.x;
+    const int half_lane = lane_id % 32;
+    const int row_offset = lane_id / 32;
+
+    const int row = blockIdx.x * 2 + row_offset;
+
+    if (row >= (int)nrows_x) return;
+
+    const uint32_t channel_dst = blockIdx.y;
+    const uint32_t channel_x   = ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
+    const uint32_t channel_y   = ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
+    const uint32_t sample_dst  = blockIdx.z;
+    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
+    const uint32_t sample_y    = sample_dst;
+
+    const int blocks_per_row = ncols_x / qk_q8_0;
+    const int kbx_offset = sample_x * stride_sample_x + channel_x * stride_channel_x + row * stride_row_x;
+
+    const block_q8_0 * x = (const block_q8_0 *)vx + kbx_offset;
+    const block_q8_1 * y = (const block_q8_1 *)vy + sample_y * stride_sample_y + channel_y * stride_channel_y;
+
+    float sumf = 0.0f;
+
+    for (int ib = half_lane; ib < blocks_per_row; ib += 32) {
+        const block_q8_0 * bq8_0 = x + ib;
+        const block_q8_1 * bq8_1 = y + ib;
+
+        // Load 32 bytes of Q8_0 quantized values (32 int8 values)
+        const int * v = (const int *)bq8_0->qs;
+        const int v0 = v[0];
+        const int v1 = v[1];
+        const int v2 = v[2];
+        const int v3 = v[3];
+        const int v4 = v[4];
+        const int v5 = v[5];
+        const int v6 = v[6];
+        const int v7 = v[7];
+
+        // Load 32 bytes of Q8_1 quantized values
+        const int * u = (const int *)bq8_1->qs;
+        const int u0 = u[0];
+        const int u1 = u[1];
+        const int u2 = u[2];
+        const int u3 = u[3];
+        const int u4 = u[4];
+        const int u5 = u[5];
+        const int u6 = u[6];
+        const int u7 = u[7];
+
+        // Compute dot product (8 dp4a for full 32 values)
+        int sumi = 0;
+        sumi = ggml_cuda_dp4a(v0, u0, sumi);
+        sumi = ggml_cuda_dp4a(v1, u1, sumi);
+        sumi = ggml_cuda_dp4a(v2, u2, sumi);
+        sumi = ggml_cuda_dp4a(v3, u3, sumi);
+        sumi = ggml_cuda_dp4a(v4, u4, sumi);
+        sumi = ggml_cuda_dp4a(v5, u5, sumi);
+        sumi = ggml_cuda_dp4a(v6, u6, sumi);
+        sumi = ggml_cuda_dp4a(v7, u7, sumi);
+
+        // Q8_0 formula: d0 * d1 * sumi (simple!)
+        const float d0 = bq8_0->d;
+        const float d1 = __low2float(bq8_1->ds);
+        sumf += d0 * d1 * (float)sumi;
+    }
+
+    // Half-warp reduction using fused DPP instructions
+    sumf = warp_reduce_sum<32>(sumf);
+
+    if (half_lane == 0) {
+        dst[sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row] = sumf;
+    }
+}
+
+static void gfx906_launch_mul_mat_vec_q8_0_warp_coop(
+        const void * vx, const void * vy, const int32_t * ids,
+        float * dst,
+        const uint32_t ncols_x, const uint3 nchannels_y,
+        const uint32_t stride_row_x,
+        const uint32_t stride_col_dst, const uint3 channel_ratio,
+        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
+        const uint32_t stride_channel_dst, const uint3 sample_ratio,
+        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
+        const uint32_t stride_sample_dst, const uint32_t nrows_x,
+        const uint32_t nchannels_dst, const uint32_t nsamples_dst,
+        cudaStream_t stream) {
+
+    const dim3 block_dims(64, 1, 1);
+    const dim3 block_nums((nrows_x + 1) / 2, nchannels_dst, nsamples_dst);
+
+    gfx906_mul_mat_vec_q8_0_warp_coop<<<block_nums, block_dims, 0, stream>>>(
+        vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x,
+        stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y,
+        stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
+        stride_sample_dst, nrows_x);
+}
+
+#endif // GGML_USE_HIP
diff --git a/ggml/src/ggml-cuda/gfx906/gfx906-vecdotq.cuh b/ggml/src/ggml-cuda/gfx906/gfx906-vecdotq.cuh
new file mode 100644
index 000000000..92f37bdef
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/gfx906-vecdotq.cuh
@@ -0,0 +1,62 @@
+#pragma once
+
+// MXFP4 dequantization using v_perm_b32 for 8-entry table lookup
+// Unaligned memory loads via memcpy (compiler optimizes to flat_load)
+
+#include "gfx906-config.h"
+
+#if defined(GGML_USE_HIP) && defined(__gfx906__)
+
+static __device__ __forceinline__ int gfx906_get_int_b1_fast(const void * x, const int & i32) {
+    const uint8_t * x8 = (const uint8_t *) x;
+    int x32;
+    memcpy(&x32, x8 + 4*i32, 4);
+    return x32;
+}
+
+static __device__ __forceinline__ int gfx906_get_int_b2_fast(const void * x, const int & i32) {
+    int x32;
+    memcpy(&x32, (const uint8_t*)x + 4*i32, 4);
+    return x32;
+}
+
+__constant__ uint8_t gfx906_mxfp4_magnitudes[8] = { 0, 1, 2, 3, 4, 6, 8, 12 };
+
+static __device__ __forceinline__ int2 gfx906_get_int_from_mxfp4_table(const uint32_t q4) {
+    const uint32_t *mags32 = (const uint32_t *)gfx906_mxfp4_magnitudes;
+
+    const uint32_t q_even = q4;
+    const uint32_t q_odd  = q4 >> 4;
+
+    uint32_t sign_even = (q_even >> 3) & 0x01010101;
+    uint32_t sign_odd  = (q_odd  >> 3) & 0x01010101;
+
+    const uint32_t sel_even = q_even & 0x07070707;
+    const uint32_t sel_odd  = q_odd  & 0x07070707;
+
+    uint32_t mag_even = __builtin_amdgcn_perm(mags32[1], mags32[0], sel_even);
+    uint32_t mag_odd  = __builtin_amdgcn_perm(mags32[1], mags32[0], sel_odd);
+
+    const uint32_t mask_even = sign_even * 0xFFu;
+    const uint32_t mask_odd  = sign_odd  * 0xFFu;
+
+    uint32_t res_x = (mag_even ^ mask_even) + sign_even;
+    uint32_t res_y = (mag_odd  ^ mask_odd)  + sign_odd;
+
+    return make_int2(res_x, res_y);
+}
+
+#define GFX906_VEC_DOT_MXFP4_Q8_1(bq4, bq8_1, iqs, sumi) \
+    do { \
+        const int * q8 = (const int *) bq8_1->qs + iqs; \
+        const int aux_q4_0 = gfx906_get_int_b1_fast(bq4->qs, iqs + 0); \
+        const int aux_q4_1 = gfx906_get_int_b1_fast(bq4->qs, iqs + 1); \
+        const int2 v0 = gfx906_get_int_from_mxfp4_table(aux_q4_0); \
+        const int2 v1 = gfx906_get_int_from_mxfp4_table(aux_q4_1); \
+        sumi = ggml_cuda_dp4a(v0.x, q8[0], sumi); \
+        sumi = ggml_cuda_dp4a(v0.y, q8[4], sumi); \
+        sumi = ggml_cuda_dp4a(v1.x, q8[1], sumi); \
+        sumi = ggml_cuda_dp4a(v1.y, q8[5], sumi); \
+    } while(0)
+
+#endif
diff --git a/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq112-dv112.cu b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq112-dv112.cu
new file mode 100644
index 000000000..7d2741ebf
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq112-dv112.cu
@@ -0,0 +1,7 @@
+// Q8 kernel template instantiation
+
+#include "../gfx906-fattn-q8.cuh"
+
+// Phase 5: Temporarily disabled - DKQ=112 is not multiple of 32 (Q8_0 block size)
+// TODO: Re-enable after adding support for non-32-multiple head sizes
+// DECL_FATTN_TILE_CASE(112, 112);
diff --git a/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq128-dv128.cu b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq128-dv128.cu
new file mode 100644
index 000000000..97bc60b0a
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq128-dv128.cu
@@ -0,0 +1,5 @@
+// Q8 kernel template instantiation
+
+#include "../gfx906-fattn-q8.cuh"
+
+DECL_FATTN_TILE_CASE(128, 128);
diff --git a/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq256-dv256.cu b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq256-dv256.cu
new file mode 100644
index 000000000..3c288b356
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq256-dv256.cu
@@ -0,0 +1,5 @@
+// Q8 kernel template instantiation
+
+#include "../gfx906-fattn-q8.cuh"
+
+DECL_FATTN_TILE_CASE(256, 256);
diff --git a/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq40-dv40.cu b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq40-dv40.cu
new file mode 100644
index 000000000..0c548f170
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq40-dv40.cu
@@ -0,0 +1,7 @@
+// Q8 kernel template instantiation
+
+#include "../gfx906-fattn-q8.cuh"
+
+// Phase 5: Temporarily disabled - DKQ=40 is not multiple of 32 (Q8_0 block size)
+// TODO: Re-enable after adding support for non-32-multiple head sizes
+// DECL_FATTN_TILE_CASE(40, 40);
diff --git a/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq576-dv512.cu b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq576-dv512.cu
new file mode 100644
index 000000000..4120d8aba
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq576-dv512.cu
@@ -0,0 +1,5 @@
+// Q8 kernel template instantiation
+
+#include "../gfx906-fattn-q8.cuh"
+
+DECL_FATTN_TILE_CASE(576, 512);
diff --git a/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq64-dv64.cu b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq64-dv64.cu
new file mode 100644
index 000000000..ed0d0773b
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq64-dv64.cu
@@ -0,0 +1,5 @@
+// Q8 kernel template instantiation
+
+#include "../gfx906-fattn-q8.cuh"
+
+DECL_FATTN_TILE_CASE(64, 64);
diff --git a/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq80-dv80.cu b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq80-dv80.cu
new file mode 100644
index 000000000..a9fe648c2
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq80-dv80.cu
@@ -0,0 +1,7 @@
+// Q8 kernel template instantiation
+
+#include "../gfx906-fattn-q8.cuh"
+
+// Phase 5: Temporarily disabled - DKQ=80 is not multiple of 32 (Q8_0 block size)
+// TODO: Re-enable after adding support for non-32-multiple head sizes
+// DECL_FATTN_TILE_CASE(80, 80);
diff --git a/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq96-dv96.cu b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq96-dv96.cu
new file mode 100644
index 000000000..788398614
--- /dev/null
+++ b/ggml/src/ggml-cuda/gfx906/template-instances/fattn-tile-q8-instance-dkq96-dv96.cu
@@ -0,0 +1,5 @@
+// Q8 kernel template instantiation
+
+#include "../gfx906-fattn-q8.cuh"
+
+DECL_FATTN_TILE_CASE(96, 96);
diff --git a/ggml/src/ggml-cuda/ggml-cuda.cu b/ggml/src/ggml-cuda/ggml-cuda.cu
index f021de1d7..4444fb1a6 100644
--- a/ggml/src/ggml-cuda/ggml-cuda.cu
+++ b/ggml/src/ggml-cuda/ggml-cuda.cu
@@ -1306,7 +1306,7 @@ static void ggml_cuda_op_mul_mat_cublas(
 
         CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
 
-        if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
+        if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_GCN(cc)) {
             const float alpha = 1.0f;
             const float beta = 0.0f;
             CUBLAS_CHECK(
@@ -1926,7 +1926,7 @@ static void ggml_cuda_mul_mat_batched_cublas_impl(ggml_backend_cuda_context & ct
 
     int id = ggml_cuda_get_device();
     const int cc = ggml_cuda_info().devices[id].cc;
-    if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
+    if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_GCN(cc)) {
         cu_compute_type = CUBLAS_COMPUTE_32F;
         alpha = &alpha_f32;
         beta = &beta_f32;
@@ -2235,9 +2235,15 @@ static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor
 
     //TODO update for generic tensor parallelism
     const int cc                 = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
-    bool use_batched_cublas_f16  = src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || !any_gpus_with_slow_fp16);
-    bool use_batched_cublas_bf16 = src0->type == GGML_TYPE_BF16 && bf16_mma_hardware_available(cc);
-    bool use_batched_cublas_f32  = src0->type == GGML_TYPE_F32;
+#ifdef GGML_HIP_NO_HIPBLASLT
+    // Disable batched cuBLAS for GCN (gfx900/gfx906) - rocBLAS lacks compatible kernels
+    const bool disable_batched_cublas = GGML_CUDA_CC_IS_GCN(cc);
+#else
+    const bool disable_batched_cublas = false;
+#endif
+    bool use_batched_cublas_f16  = !disable_batched_cublas && src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || !any_gpus_with_slow_fp16);
+    bool use_batched_cublas_bf16 = !disable_batched_cublas && src0->type == GGML_TYPE_BF16 && bf16_mma_hardware_available(cc);
+    bool use_batched_cublas_f32  = !disable_batched_cublas && src0->type == GGML_TYPE_F32;
 
     if (!split && use_mul_mat_vec_f) {
         // the custom F16 vector kernel can be used over batched cuBLAS GEMM
@@ -2726,6 +2732,10 @@ static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct gg
         case GGML_OP_RWKV_WKV7:
             ggml_cuda_op_rwkv_wkv7(ctx, dst);
             break;
+        case GGML_OP_DELTANET:
+            // ggml_cuda_op_deltanet(ctx, dst);  // Uncomment in Phase 2
+            GGML_ABORT("DELTANET not yet implemented");
+            break;
         case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
             ggml_cuda_cross_entropy_loss_back(ctx, dst);
             break;
@@ -2853,9 +2863,9 @@ static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
 }
 
 #ifdef USE_CUDA_GRAPH
-static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {
+static bool check_node_graph_compatibility(ggml_cgraph * cgraph,
+    bool use_cuda_graph) {
 
-    bool use_cuda_graph = true;
     // Loop over nodes in GGML graph to obtain info needed for CUDA graph
 
     const std::string gemma3n_per_layer_proj_src0_name = "inp_per_layer_selected";
@@ -2915,41 +2925,41 @@ static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {
     return use_cuda_graph;
 }
 
-static void ggml_cuda_graph_node_set_properties(ggml_cuda_graph_node_properties * props, ggml_tensor * node) {
-    props->node_address = node->data;
-    props->node_op = node->op;
+static void set_ggml_graph_node_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
+    graph_node_properties->node_address = node->data;
+    graph_node_properties->node_op = node->op;
     for (int i = 0; i < GGML_MAX_DIMS; i++) {
-        props->ne[i] = node->ne[i];
-        props->nb[i] = node->nb[i];
+        graph_node_properties->ne[i] = node->ne[i];
+        graph_node_properties->nb[i] = node->nb[i];
     }
     for (int i = 0; i < GGML_MAX_SRC; i++) {
-        props->src_address[i] = node->src[i] ? node->src[i]->data : nullptr;
+        graph_node_properties->src_address[i] = node->src[i] ? node->src[i]->data : nullptr;
     }
-    memcpy(props->op_params, node->op_params, GGML_MAX_OP_PARAMS);
+    memcpy(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS);
 }
 
-static bool ggml_cuda_graph_node_properties_match(ggml_tensor * node, ggml_cuda_graph_node_properties * props) {
-    if (node->data != props->node_address &&
+static bool ggml_graph_node_has_matching_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
+    if (node->data != graph_node_properties->node_address &&
           node->op != GGML_OP_VIEW) {
         return false;
     }
 
-    if (node->op != props->node_op) {
+    if (node->op != graph_node_properties->node_op) {
         return false;
     }
 
     for (int i = 0; i < GGML_MAX_DIMS; i++) {
-        if (node->ne[i] != props->ne[i]) {
+        if (node->ne[i] != graph_node_properties->ne[i]) {
             return false;
         }
-        if (node->nb[i] != props->nb[i]) {
+        if (node->nb[i] != graph_node_properties->nb[i]) {
             return false;
         }
     }
 
     for (int i = 0; i < GGML_MAX_SRC; i++) {
         if (node->src[i] &&
-            node->src[i]->data != props->src_address[i] &&
+            node->src[i]->data != graph_node_properties->src_address[i] &&
             node->op != GGML_OP_VIEW
         ) {
             return false;
@@ -2957,55 +2967,44 @@ static bool ggml_cuda_graph_node_properties_match(ggml_tensor * node, ggml_cuda_
     }
 
     if ((node->op == GGML_OP_SCALE || node->op == GGML_OP_GLU) &&
-        memcmp(props->op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
+        memcmp(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
         return false;
     }
 
     return true;
 }
 
-static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
+static bool is_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
 
-    bool res = false;
+    bool cuda_graph_update_required = false;
 
     if (cuda_ctx->cuda_graph->instance == nullptr) {
-        res = true;
+        cuda_graph_update_required = true;
     }
 
     // Check if the graph size has changed
-    if (cuda_ctx->cuda_graph->props.size() != (size_t)cgraph->n_nodes + cgraph->n_leafs) {
-        res = true;
-        cuda_ctx->cuda_graph->props.resize(cgraph->n_nodes + cgraph->n_leafs);
+    if (cuda_ctx->cuda_graph->ggml_graph_properties.size() != (size_t)cgraph->n_nodes) {
+        cuda_graph_update_required = true;
+        cuda_ctx->cuda_graph->ggml_graph_properties.resize(cgraph->n_nodes);
     }
 
     // Loop over nodes in GGML graph to determine if CUDA graph update is required
     // and store properties to allow this comparison for the next token
     for (int i = 0; i < cgraph->n_nodes; i++) {
-        bool props_match = true;
-        if (!res) {
-            props_match = ggml_cuda_graph_node_properties_match(cgraph->nodes[i], &cuda_ctx->cuda_graph->props[i]);
+        bool has_matching_properties = true;
+        if (!cuda_graph_update_required) {
+            has_matching_properties = ggml_graph_node_has_matching_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
         }
-        if (!props_match) {
-            res = true;
+        if (!has_matching_properties) {
+            cuda_graph_update_required = true;
         }
-        ggml_cuda_graph_node_set_properties(&cuda_ctx->cuda_graph->props[i], cgraph->nodes[i]);
+        set_ggml_graph_node_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
     }
 
-    for (int i = 0; i < cgraph->n_leafs; i++) {
-        bool props_match= true;
-        if (!res) {
-            props_match = ggml_cuda_graph_node_properties_match(cgraph->leafs[i], &cuda_ctx->cuda_graph->props[cgraph->n_nodes + i]);
-        }
-        if (!props_match) {
-            res = true;
-        }
-        ggml_cuda_graph_node_set_properties(&cuda_ctx->cuda_graph->props[cgraph->n_nodes + i], cgraph->leafs[i]);
-    }
-
-    return res;
+    return cuda_graph_update_required;
 }
 
-static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx) {
+static void update_cuda_graph_executable(ggml_backend_cuda_context * cuda_ctx) {
 
 #if CUDART_VERSION >= 12000
     cudaGraphExecUpdateResultInfo result_info;
@@ -3236,11 +3235,10 @@ static bool ggml_cuda_can_fuse(const struct ggml_cgraph * cgraph, int node_idx,
     return false;
 }
 
-static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required) {
-    bool graph_evaluated_or_captured = false;
-
+static void evaluate_and_capture_cuda_graph(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
+    bool & graph_evaluated_or_captured, bool & use_cuda_graph, bool & cuda_graph_update_required) {
     // flag used to determine whether it is an integrated_gpu
-    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;
+    const bool integrated = ggml_cuda_info().devices[cuda_ctx->device].integrated;
 
     ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
     bool                         is_concurrent_event_active = false;
@@ -3261,7 +3259,7 @@ static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cud
 
             for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                 cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
-                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
+                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event, 0));
             }
         }
     };
@@ -3345,7 +3343,7 @@ static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cud
                             // Wait on join events of forked streams in the main stream
                             CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                        cuda_ctx->stream(cuda_ctx->device, i)));
-                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
+                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1], 0));
                         }
 
                         is_concurrent_event_active = false;
@@ -3710,7 +3708,7 @@ static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cud
             CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
         }
         if (cuda_graph_update_required) { // Update graph executable
-            ggml_cuda_graph_update_executable(cuda_ctx);
+            update_cuda_graph_executable(cuda_ctx);
         }
         // Launch graph
         CUDA_CHECK(cudaGraphLaunch(cuda_ctx->cuda_graph->instance, cuda_ctx->stream()));
@@ -3720,25 +3718,45 @@ static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cud
     }
 }
 
-static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx) {
+static bool ggml_cuda_set_cuda_graph_enabled(ggml_backend_cuda_context * cuda_ctx) {
 
 #ifdef USE_CUDA_GRAPH
+    static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);
 
+    // Objects required for CUDA Graph
     if (cuda_ctx->cuda_graph == nullptr) {
         cuda_ctx->cuda_graph.reset(new ggml_cuda_graph());
     }
 
+    bool use_cuda_graph = true;
+
     if (cuda_ctx->cuda_graph->graph == nullptr) {
         if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_AMPERE) {
             cuda_ctx->cuda_graph->disable_due_to_gpu_arch = true;
+#ifndef NDEBUG
             GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
+#endif
         }
     }
 
-    return cuda_ctx->cuda_graph->is_enabled();
+    // Disable CUDA graphs in presence of env var, old GPU, use-case which is changing too rapidly,
+    // or previous graph capture failure.
+    // Also disable for multi-gpu for now. TO DO investigate
+    if (disable_cuda_graphs_due_to_env
+        || cuda_ctx->cuda_graph->disable_due_to_gpu_arch
+        || cuda_ctx->cuda_graph->disable_due_to_too_many_updates
+        || cuda_ctx->cuda_graph->disable_due_to_failed_graph_capture) {
+        use_cuda_graph = false;
+    }
+
+    cuda_ctx->cuda_graph->cuda_graphs_enabled = use_cuda_graph;
 #else
-    return false;
+
+    bool use_cuda_graph = false;
+
 #endif // USE_CUDA_GRAPH
+
+    return use_cuda_graph;
 }
 
 static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
@@ -3749,14 +3767,30 @@ static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend,
     bool use_cuda_graph             = false;
     bool cuda_graph_update_required = false;
 
+    // graph_optimize calls set_cuda_graph_enabled, in-case it not called (i.e. graph_compute is directly called)
+    // we call it here instead.
 #ifdef USE_CUDA_GRAPH
-    use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx);
+    use_cuda_graph = ggml_cuda_set_cuda_graph_enabled(cuda_ctx);
+
+    if (use_cuda_graph) {
+        cuda_graph_update_required = is_cuda_graph_update_required(cuda_ctx, cgraph);
 
-    if (cuda_ctx->cuda_graph->is_enabled()) {
-        cuda_graph_update_required = ggml_cuda_graph_update_required(cuda_ctx, cgraph);
-        use_cuda_graph             = ggml_cuda_graph_check_compability(cgraph);
+        use_cuda_graph = check_node_graph_compatibility(cgraph, use_cuda_graph);
 
-        cuda_ctx->cuda_graph->record_update(use_cuda_graph, cuda_graph_update_required);
+        // Disable CUDA graphs (from the next token) if the use-case is demanding too many consecutive graph updates.
+        if (use_cuda_graph && cuda_graph_update_required) {
+            cuda_ctx->cuda_graph->number_consecutive_updates++;
+        } else {
+            cuda_ctx->cuda_graph->number_consecutive_updates = 0;
+        }
+
+        if (cuda_ctx->cuda_graph->number_consecutive_updates >= 4) {
+            cuda_ctx->cuda_graph->disable_due_to_too_many_updates = true;
+            cuda_ctx->cuda_graph->cuda_graphs_enabled = false;
+#ifndef NDEBUG
+            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
+#endif
+        }
     }
 #endif // USE_CUDA_GRAPH
 
@@ -3770,7 +3804,9 @@ static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend,
         CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
     }
 
-    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required);
+    bool graph_evaluated_or_captured = false;
+
+    evaluate_and_capture_cuda_graph(cuda_ctx, cgraph, graph_evaluated_or_captured, use_cuda_graph, cuda_graph_update_required);
 
     return GGML_STATUS_SUCCESS;
 }
@@ -3803,7 +3839,7 @@ static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_ev
 static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph) {
     ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
 
-    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx);
+    const bool use_cuda_graph = ggml_cuda_set_cuda_graph_enabled(cuda_ctx);
 
     static bool enable_graph_optimization = [] {
         const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
@@ -4122,7 +4158,6 @@ struct ggml_backend_cuda_device_context {
     std::string name;
     std::string description;
     std::string pci_bus_id;
-    int op_offload_min_batch_size;
 };
 
 static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
@@ -4647,8 +4682,12 @@ static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const g
         case GGML_OP_CUMSUM:
         case GGML_OP_TRI:
         case GGML_OP_DIAG:
-        case GGML_OP_SOLVE_TRI:
             return true;
+        case GGML_OP_SOLVE_TRI:
+            // GFX906: limit SOLVE_TRI dimensions to avoid hipBLASLt crashes
+            return op->src[0]->ne[0] <= 64 && op->src[1]->ne[0] <= 32;
+        case GGML_OP_DELTANET:
+            return false;  // Change to true in Phase 2
 
         default:
             return false;
@@ -4677,9 +4716,11 @@ static int64_t get_op_batch_size(const ggml_tensor * op) {
 }
 
 static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
-    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
+    const int min_batch_size = 32;
 
-    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
+    return get_op_batch_size(op) >= min_batch_size;
+
+    GGML_UNUSED(dev);
 }
 
 static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
@@ -4847,7 +4888,6 @@ ggml_backend_reg_t ggml_backend_cuda_reg() {
         std::lock_guard<std::mutex> lock(mutex);
         if (!initialized) {
             ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
-            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
 
             for (int i = 0; i < ggml_cuda_info().device_count; i++) {
                 ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
@@ -4861,7 +4901,6 @@ ggml_backend_reg_t ggml_backend_cuda_reg() {
                 char pci_bus_id[16] = {};
                 snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
                 dev_ctx->pci_bus_id = pci_bus_id;
-                dev_ctx->op_offload_min_batch_size = min_batch_size;
 
                 ggml_backend_dev_t dev = new ggml_backend_device {
                     /* .iface   = */ ggml_backend_cuda_device_interface,
diff --git a/ggml/src/ggml-cuda/mean.cu b/ggml/src/ggml-cuda/mean.cu
index 60542fc19..691d8dcb1 100644
--- a/ggml/src/ggml-cuda/mean.cu
+++ b/ggml/src/ggml-cuda/mean.cu
@@ -34,11 +34,13 @@ void ggml_cuda_op_mean(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
             // CUDA_GRAPHS_DISABLED
             ((ncols > 65536) &&
              ((ctx.cuda_graph->instance == nullptr) && (iscapturing == cudaStreamCaptureStatusNone) ||
-              ctx.cuda_graph->is_enabled())) ||
+              ctx.cuda_graph->disable_due_to_gpu_arch || ctx.cuda_graph->disable_due_to_too_many_updates ||
+              ctx.cuda_graph->disable_due_to_failed_graph_capture)) ||
         // CUDA_GRAPHS ENABLED
         ((ncols > 32768) &&
          !((ctx.cuda_graph->instance == nullptr) && (iscapturing == cudaStreamCaptureStatusNone) ||
-            ctx.cuda_graph->is_enabled()))) {
+           ctx.cuda_graph->disable_due_to_gpu_arch || ctx.cuda_graph->disable_due_to_too_many_updates ||
+           ctx.cuda_graph->disable_due_to_failed_graph_capture))) {
 #else
         (ncols > 65536)) {
 #endif // USE_CUDA_GRAPH
diff --git a/ggml/src/ggml-cuda/mma.cuh b/ggml/src/ggml-cuda/mma.cuh
index df9eed711..42085d100 100644
--- a/ggml/src/ggml-cuda/mma.cuh
+++ b/ggml/src/ggml-cuda/mma.cuh
@@ -206,10 +206,16 @@ namespace ggml_cuda_mma {
 
         static __device__ __forceinline__ int get_j(const int l) {
             if constexpr (I == 16 && J == 16) {
-                // matrix C
 #if defined(RDNA3)
-                return 2 * l + (threadIdx.x / 16);
+                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>) {
+                    // matrix C
+                    return 2 * l + (threadIdx.x / 16);
+                } else {
+                    // matrix A&B
+                    return l;
+                }
 #else
+                // matrix C is the transposed matrix A&B on RDNA4
                 return ne * (threadIdx.x / 16) + l;
 #endif // defined(RDNA3)
             } else if constexpr (I == 16 && J == 8) {
@@ -621,6 +627,21 @@ namespace ggml_cuda_mma {
 
         return ret;
     }
+#elif defined(AMD_WMMA_AVAILABLE)
+    template <int I, int J>
+    static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
+        tile<I, J/2, half2> ret;
+#pragma unroll
+        for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
+            ret.x[l0/2] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
+        }
+        return ret;
+    }
+
+    static __device__ __forceinline__ tile<8, 8, half2> get_transposed(const tile<16, 4, half2> & t) {
+        NO_DEVICE_CODE;
+        return tile<8, 8, half2>{};
+    }
 #else // Volta
     template <int I, int J>
     static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
@@ -639,6 +660,19 @@ namespace ggml_cuda_mma {
     }
 #endif // defined(TURING_MMA_AVAILABLE)
 
+    static __device__ __forceinline__ void make_identity_mat(tile<16, 8, half2> & t) {
+#if defined(RDNA4)
+        const int row = t.get_i(0);
+        const int left_right = t.get_j(0) / 4;
+        const int up_down = row / 8;
+        const int idx = row % 8;
+        reinterpret_cast<half*>(t.x)[idx] = left_right == up_down ? 1.0f : 0.0f;
+#else
+        GGML_UNUSED_VARS(t);
+        NO_DEVICE_CODE;
+#endif // defined(RDNA4)
+    }
+
     template <int I, int J, typename T, data_layout dl>
     static __device__ __forceinline__ void load_generic(tile<I, J, T, dl> & t, const T * __restrict__ xs0, const int stride) {
 #if defined(AMD_MFMA_AVAILABLE)
@@ -878,6 +912,17 @@ namespace ggml_cuda_mma {
             : "+r"(Dxi[2]), "+r"(Dxi[3])
             : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[3]));
 #endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
+#elif defined(AMD_WMMA_AVAILABLE)
+#if defined(RDNA4)
+        using halfx8_t = __attribute__((ext_vector_type(8))) _Float16;
+        halfx8_t& acc_frag = reinterpret_cast<halfx8_t&>(D.x[0]);
+        const halfx8_t& a_frag = reinterpret_cast<const halfx8_t&>(A.x[0]);
+        const halfx8_t& b_frag = reinterpret_cast<const halfx8_t&>(B.x[0]);
+        acc_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a_frag, b_frag, acc_frag);
+#else
+        GGML_UNUSED_VARS(D, A, B);
+        NO_DEVICE_CODE;
+#endif // defined(RDNA4)
 #else
         GGML_UNUSED_VARS(D, A, B);
         NO_DEVICE_CODE;
diff --git a/ggml/src/ggml-cuda/mmid.cu b/ggml/src/ggml-cuda/mmid.cu
index 3c61e4595..601e745bb 100644
--- a/ggml/src/ggml-cuda/mmid.cu
+++ b/ggml/src/ggml-cuda/mmid.cu
@@ -138,6 +138,19 @@ static void launch_mm_ids_helper(
 void ggml_cuda_launch_mm_ids_helper(
         const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
         const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
+
+#if defined(GGML_USE_HIP)
+    // On AMD wavefront64 GPUs (like MI50/gfx906), the optimized paths use sub-warp shuffles
+    // that don't work correctly when n_expert_used >= warp_size/2 (the sub-warp width).
+    // Fall back to generic path only for these cases.
+    const int id = ggml_cuda_get_device();
+    const int warp_size = ggml_cuda_info().devices[id].warp_size;
+    if (n_expert_used >= warp_size / 2) {
+        launch_mm_ids_helper<0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
+        return;
+    }
+#endif
+
     switch (n_expert_used) {
         case  2:
             launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
diff --git a/ggml/src/ggml-cuda/mmq.cu b/ggml/src/ggml-cuda/mmq.cu
index ceb95758d..9a69f41d1 100644
--- a/ggml/src/ggml-cuda/mmq.cu
+++ b/ggml/src/ggml-cuda/mmq.cu
@@ -190,7 +190,7 @@ void ggml_cuda_mul_mat_q(
     {
         const int64_t s11 = src1->nb[1] / ts_src1;
         const int64_t s12 = src1->nb[2] / ts_src1;
-        const int64_t s13 = src1->nb[2] / ts_src1;
+        const int64_t s13 = src1->nb[3] / ts_src1;
 
         if (use_native_mxfp4) {
             quantize_mmq_mxfp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
@@ -333,28 +333,31 @@ bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t
     }
 
     if (amd_wmma_available(cc)) {
-        // RDNA 4 is consistently worse on rocblas
-        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
         if (GGML_CUDA_CC_IS_RDNA3(cc)) {
-            // High expert counts almost always better on MMQ
-            // due to a large amount of graph splits
+            // High expert counts are almost always better on MMQ due to
+            //     the synchronization overhead in the cuBLAS/hipBLAS path:
             // https://github.com/ggml-org/llama.cpp/pull/18202
             if (n_experts >= 64) {
                 return true;
             }
 
+            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
+            //     so it's only faster for sufficiently small batch sizes:
             switch (type) {
-                // These quants are really bad on MMQ
                 case GGML_TYPE_Q2_K:
+                    return ne11 <= 128;
                 case GGML_TYPE_Q6_K:
-                // These quants are usually worse but not always
+                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                 case GGML_TYPE_IQ2_XS:
                 case GGML_TYPE_IQ2_S:
-                    return ne11 <= 128;
+                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                 default:
                     return true;
             }
         }
+
+        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
+        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
         return true;
     }
 
diff --git a/ggml/src/ggml-cuda/mmq.cuh b/ggml/src/ggml-cuda/mmq.cuh
index a382e6a69..d2b167da0 100644
--- a/ggml/src/ggml-cuda/mmq.cuh
+++ b/ggml/src/ggml-cuda/mmq.cuh
@@ -9,10 +9,24 @@
 
 using namespace ggml_cuda_mma;
 
+// GFX906 MMQ optimizations (vectorized loads and prefetch)
+#ifdef GGML_USE_HIP
+    #include "gfx906/gfx906-mmq.cuh"
+    #include "gfx906/gfx906-config.h"
+    #include "gfx906/gfx906-mmq-prefetch.cuh"
+#endif
+
 #define MMQ_DP4A_MAX_BATCH_SIZE 64 // Max. batch size to use for dp4a MMQ kernels when FP16 tensor cores are available.
-#define MMQ_ITER_K 256
-#define MMQ_ITER_K_MXFP4_FP4    512
-#define MMQ_NWARPS 8
+
+// GFX906-optimized MMQ configuration
+#ifdef GGML_USE_HIP
+    #define MMQ_ITER_K GFX906_MMQ_ITER_K
+    #define MMQ_NWARPS GFX906_MMQ_NWARPS
+#else
+    #define MMQ_ITER_K 256
+    #define MMQ_NWARPS 8
+#endif
+#define MMQ_ITER_K_MXFP4_FP4 512
 
 typedef void (*load_tiles_mmq_t)(const char * __restrict__ x, int * x_tile, const int kbx0, const int i_max, const int stride);
 typedef void (*vec_dot_mmq_t)(const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00);
@@ -251,6 +265,15 @@ static constexpr __host__ __device__ int mmq_get_mma_tile_x_k(ggml_type type) {
 #define MMQ_TILE_Y_K     (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)
 #define MMQ_TILE_Y_FP4_K MMQ_TILE_Y_K
 
+// LDS stride for Y-tile - PADDING ANALYSIS RESULTS:
+// Original stride 40: 40 mod 32 = 8 → 4-way bank conflicts (9.3% LDS stalls)
+// Tested: Padded stride 41 with dst_idx = l + l/40 mapping
+// Result: -3.5% slower (1180 vs 1223 t/s) even with proper shared mem allocation
+// Root cause: Division overhead in store loop outweighs bank conflict reduction
+// The bank conflicts occur during vec_dot reads, but the overhead is in stores
+// Conclusion: Keep original stride - bank conflicts are cheaper than index math
+#define MMQ_TILE_Y_K_LDS MMQ_TILE_Y_K
+
 static int mmq_get_granularity_host(const int mmq_x, const int cc) {
     if (amd_mfma_available(cc) || amd_wmma_available(cc)) {
         return mmq_x >= 128 ? 32 : 16;
@@ -384,15 +407,20 @@ static __device__ __forceinline__ void vec_dot_q4_0_q8_1_dp4a(
 
                 int u[2*VDR_Q4_0_Q8_1_MMQ];
 
+#if defined(GGML_USE_HIP)
+                // Call GFX906-optimized vectorized load from gfx906/gfx906-mmq.cuh
+                gfx906_load_q4_0_quants_vectorized(y_qs, j*MMQ_TILE_Y_K_LDS + kyqs, QI4_0, u);
+#else
 #pragma unroll
                 for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
-                    u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + kyqs +  l];
-                    u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + kyqs + (l + QI4_0)];
+                    u[2*l+0] = y_qs[j*MMQ_TILE_Y_K_LDS + kyqs +  l];
+                    u[2*l+1] = y_qs[j*MMQ_TILE_Y_K_LDS + kyqs + (l + QI4_0)];
                 }
+#endif
 
                 sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
                     (&x_qs[i*(MMQ_TILE_NE_K + 1) + k0/QR4_0], u,
-                     x_df[i*(MMQ_TILE_NE_K/QI4_0) + i/QI4_0 + k0/(QR4_0*QI4_0)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
+                     x_df[i*(MMQ_TILE_NE_K/QI4_0) + i/QI4_0 + k0/(QR4_0*QI4_0)], y_ds[j*MMQ_TILE_Y_K_LDS + k01/QI8_1]);
             }
         }
     }
@@ -487,15 +515,20 @@ static __device__ __forceinline__ void vec_dot_q4_1_q8_1_dp4a(
 
                 int u[2*VDR_Q4_1_Q8_1_MMQ];
 
+#if defined(GGML_USE_HIP)
+                // Call GFX906-optimized vectorized load from gfx906/gfx906-mmq.cuh
+                gfx906_load_q4_1_quants_vectorized(y_qs, j*MMQ_TILE_Y_K_LDS + kyqs, QI4_1, u);
+#else
 #pragma unroll
                 for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
-                    u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + kyqs +  l];
-                    u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + kyqs + (l + QI4_1)];
+                    u[2*l+0] = y_qs[j*MMQ_TILE_Y_K_LDS + kyqs +  l];
+                    u[2*l+1] = y_qs[j*MMQ_TILE_Y_K_LDS + kyqs + (l + QI4_1)];
                 }
+#endif
 
                 sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
                     (&x_qs[i*(MMQ_TILE_NE_K + 1) + k0/QR4_1], u,
-                     x_dm[i*(MMQ_TILE_NE_K/QI4_1) + i/QI4_1 + k0/(QR4_1*QI4_1)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
+                     x_dm[i*(MMQ_TILE_NE_K/QI4_1) + i/QI4_1 + k0/(QR4_1*QI4_1)], y_ds[j*MMQ_TILE_Y_K_LDS + k01/QI8_1]);
             }
         }
     }
@@ -676,24 +709,46 @@ template <int mmq_y, bool need_check> static __device__ __forceinline__ void loa
     const int kbx  = txi / QI8_0;
     const int kqsx = txi % QI8_0;
 
+#if defined(GGML_USE_HIP) && defined(__gfx906__)
+    // GFX906: Software pipelining using macros from gfx906-mmq.cuh
+    constexpr int loop_iters = mmq_y / (nrows * nwarps);
+    constexpr int cache_size = loop_iters > 16 ? 16 : loop_iters;
+    int qs0_cache[cache_size];
+    int qs1_cache[cache_size];
+    int i_slot_cache[cache_size];
+
+    // Load all data into registers (async)
+    GFX906_LOAD_TILES_Q8_0_ASYNC(cache_size, nrows, nwarps, threads_per_row, need_check,
+        x, kbx0, stride, i_max, txi, kbx, kqsx, qs0_cache, qs1_cache, i_slot_cache);
+
+    // Store all to LDS
+#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
+    GFX906_STORE_TILES_Q8_0_LDS_MMA(cache_size, x_qs, qs0_cache, qs1_cache, i_slot_cache, txi);
+#else
+    GFX906_STORE_TILES_Q8_0_LDS_LEGACY(cache_size, x_qs, qs0_cache, qs1_cache, i_slot_cache, txi);
+#endif
+#else
 #pragma unroll
     for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
-        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
-
-        if (need_check) {
-            i = min(i, i_max);
-        }
+        // GFX906 optimization: Avoid LDS write conflicts in need_check path.
+        // Original code clamped i to i_max, causing all out-of-bounds threads to
+        // write to the SAME location (tile[i_max*...]) - serializing LDS writes.
+        // Fix: Each thread writes to its ORIGINAL slot; out-of-bounds write zeros.
+        const int i_slot = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
+        const int i_read = need_check ? min(i_slot, i_max) : i_slot;
+        const bool oob = need_check && (i_slot > i_max);
 
-        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbx;
+        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i_read*stride + kbx;
 
 #if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
-        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 0             + txi] = get_int_b2(bxi[0].qs,                   kqsx);
-        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
+        x_qs[i_slot*MMQ_MMA_TILE_X_K_Q8_0 + 0             + txi] = oob ? 0 : get_int_b2(bxi[0].qs,                   kqsx);
+        x_qs[i_slot*MMQ_MMA_TILE_X_K_Q8_0 + MMQ_TILE_NE_K + txi] = oob ? 0 : get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
 #else
-        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 0             + txi] = get_int_b2(bxi[0].qs,                   kqsx);
-        x_qs[i*(2*MMQ_TILE_NE_K + 1) + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
+        x_qs[i_slot*(2*MMQ_TILE_NE_K + 1) + 0             + txi] = oob ? 0 : get_int_b2(bxi[0].qs,                   kqsx);
+        x_qs[i_slot*(2*MMQ_TILE_NE_K + 1) + MMQ_TILE_NE_K + txi] = oob ? 0 : get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
 #endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
     }
+#endif
 
     constexpr int blocks_per_tile_x_row = 2*MMQ_TILE_NE_K / QI8_0;
     constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
@@ -701,18 +756,17 @@ template <int mmq_y, bool need_check> static __device__ __forceinline__ void loa
 
 #pragma unroll
     for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
-        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
-
-        if (need_check) {
-            i = min(i, i_max);
-        }
+        // Same optimization for scale loading
+        const int i_slot = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
+        const int i_read = need_check ? min(i_slot, i_max) : i_slot;
+        const bool oob = need_check && (i_slot > i_max);
 
-        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbxd;
+        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i_read*stride + kbxd;
 
 #if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
-        x_df[i*MMQ_MMA_TILE_X_K_Q8_0                 + kbxd] = bxi->d;
+        x_df[i_slot*MMQ_MMA_TILE_X_K_Q8_0                 + kbxd] = oob ? 0.0f : (float)bxi->d;
 #else
-        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + kbxd] = bxi->d;
+        x_df[i_slot*(2*MMQ_TILE_NE_K/QI8_0) + i_slot/(QI8_0/2) + kbxd] = oob ? 0.0f : (float)bxi->d;
 #endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
     }
 }
@@ -737,6 +791,41 @@ template <int mmq_y, bool need_check> static __device__ __forceinline__ void loa
     const int kbx  = txi / QI_MXFP4;
     const int kqsx = txi % QI_MXFP4;
 
+#if defined(GGML_USE_HIP) && defined(__gfx906__)
+    // GFX906: Software pipelining - load all data first, then dequant all
+    // Maximizes memory-level parallelism before compute
+    constexpr int loop_iters = mmq_y / (nrows * nwarps);
+    int aux_q4_cache[loop_iters > 16 ? 16 : loop_iters];
+    int i_cache[loop_iters > 16 ? 16 : loop_iters];
+
+    // Phase 1: Issue all loads
+    #pragma unroll
+    for (int iter = 0; iter < (loop_iters > 16 ? 16 : loop_iters); iter++) {
+        const int i0 = iter * nrows * nwarps;
+        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
+        if (need_check) {
+            i = min(i, i_max);
+        }
+        const block_mxfp4 * bxi = (const block_mxfp4 *) x + kbx0 + i*stride + kbx;
+        aux_q4_cache[iter] = get_int_b1(bxi->qs, kqsx);
+        i_cache[iter] = i;
+    }
+
+    // Phase 2: Dequant and store
+    const int k0 = kbx * (2 * QI_MXFP4) + kqsx;
+    #pragma unroll
+    for (int iter = 0; iter < (loop_iters > 16 ? 16 : loop_iters); iter++) {
+        const int2 v = get_int_from_mxfp4_table(aux_q4_cache[iter]);
+        const int i = i_cache[iter];
+#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
+        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + k0 + 0]        = v.x;
+        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + k0 + QI_MXFP4] = v.y;
+#else
+        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 0]        = v.x;
+        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + QI_MXFP4] = v.y;
+#endif
+    }
+#else
 #pragma unroll
     for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
         int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
@@ -748,7 +837,7 @@ template <int mmq_y, bool need_check> static __device__ __forceinline__ void loa
         const block_mxfp4 * bxi = (const block_mxfp4 *) x + kbx0 + i*stride + kbx;
 
         const int aux_q4 = get_int_b1(bxi->qs, kqsx);
-        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
+        const int2 v = get_int_from_mxfp4_table(aux_q4);
         const int k0 = kbx * (2 * QI_MXFP4) + kqsx;
 
 #if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
@@ -759,6 +848,7 @@ template <int mmq_y, bool need_check> static __device__ __forceinline__ void loa
         x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + QI_MXFP4] = v.y;
 #endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
     }
+#endif
 
     constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI_MXFP4;
     constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
@@ -851,8 +941,8 @@ static __device__ __forceinline__ void vec_dot_q8_0_q8_1_dp4a(
                 const int i = i0 + threadIdx.x;
 
                 sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>
-                    (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k0 % MMQ_TILE_NE_K],
-                     x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + k0/QI8_0], y_df[j*MMQ_TILE_Y_K + (k0/QI8_1) % (MMQ_TILE_NE_K/QI8_1)]);
+                    (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K_LDS + k0 % MMQ_TILE_NE_K],
+                     x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + k0/QI8_0], y_df[j*MMQ_TILE_Y_K_LDS + (k0/QI8_1) % (MMQ_TILE_NE_K/QI8_1)]);
             }
         }
     }
@@ -871,7 +961,7 @@ static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
     constexpr int rows_per_warp = granularity;
     constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
-    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
+    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K_LDS);
 
     const int   * x_qs = (const int   *) x;
     const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
@@ -893,14 +983,14 @@ static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
 #pragma unroll
         for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
             tile_B B;
-            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
+            load_generic(B, y_qs + j0*MMQ_TILE_Y_K_LDS + k01, MMQ_TILE_Y_K_LDS);
 
             float dB;
             const int j = j0 + tile_C::get_j(0);
             if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
-                dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
+                dB = y_df[j*MMQ_TILE_Y_K_LDS + k01/QI8_1];
             } else {
-                dB = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
+                dB = __low2float(y_ds[j*MMQ_TILE_Y_K_LDS + k01/QI8_1]);
             }
 
 #pragma unroll
@@ -926,7 +1016,7 @@ static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
     constexpr int rows_per_warp = 2 * granularity;
     constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
-    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
+    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K_LDS);
 
     const int   * x_qs = (const int   *) x;
     const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
@@ -968,16 +1058,16 @@ static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
             tile_B B;
             float dB[tile_C::ne/2];
 
-            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix
+            load_generic(B, y_qs + j0*MMQ_TILE_Y_K_LDS + k01, MMQ_TILE_Y_K_LDS); // faster than load_ldmatrix
 
 #pragma unroll
             for (int l = 0; l < tile_C::ne/2; ++l) {
                 const int j = j0 + tile_C::get_j(l);
 
                 if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
-                    dB[l] =             y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
+                    dB[l] =             y_df[j*MMQ_TILE_Y_K_LDS + k01/QI8_1];
                 } else {
-                    dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
+                    dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K_LDS + k01/QI8_1]);
                 }
             }
 
@@ -1093,8 +1183,8 @@ static __device__ __forceinline__ void vec_dot_q8_1_q8_1_dp4a(
                 const int i = i0 + threadIdx.x;
 
                 sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
-                    (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
-                    x_dm[i*(MMQ_TILE_NE_K/QI5_1) + i/QI5_1 + k0/QI8_1], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
+                    (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K_LDS + k01],
+                    x_dm[i*(MMQ_TILE_NE_K/QI5_1) + i/QI5_1 + k0/QI8_1], y_ds[j*MMQ_TILE_Y_K_LDS + k01/QI8_1]);
             }
         }
     }
@@ -1113,7 +1203,7 @@ static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
     constexpr int rows_per_warp = granularity;
     constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
-    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
+    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K_LDS);
 
     const int   * x_qs = (const int   *) x;
     const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
@@ -1134,10 +1224,10 @@ static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
 #pragma unroll
         for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
             tile_B B;
-            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
+            load_generic(B, y_qs + j0*MMQ_TILE_Y_K_LDS + k01, MMQ_TILE_Y_K_LDS);
 
             const int j = j0 + tile_C::get_j(0);
-            const float2 dsB = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
+            const float2 dsB = __half22float2(y_dm[j*MMQ_TILE_Y_K_LDS + k01/QI8_1]);
 
 #pragma unroll
             for (int n = 0; n < ntx; ++n) {
@@ -1163,7 +1253,7 @@ static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
     constexpr int rows_per_warp = 2 * granularity;
     constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
-    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
+    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K_LDS);
 
     const int   * x_qs = (const int   *) x;
     const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
@@ -1204,13 +1294,13 @@ static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
             tile_B   B;
             float2 dsB[tile_C::ne/2];
 
-            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix
+            load_generic(B, y_qs + j0*MMQ_TILE_Y_K_LDS + k01, MMQ_TILE_Y_K_LDS); // faster than load_ldmatrix
 
 #pragma unroll
             for (int l = 0; l < tile_C::ne/2; ++l) {
                 const int j = j0 + tile_C::get_j(l);
 
-                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
+                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K_LDS + k01/QI8_1]);
             }
 
 #pragma unroll
@@ -1256,9 +1346,9 @@ static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_dp4a(
 
                 sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q8_0_16_q8_1_impl<QI8_0>(
                     &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0],
-                    &y_qs[j*MMQ_TILE_Y_K + k01],
+                    &y_qs[j*MMQ_TILE_Y_K_LDS + k01],
                     &x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + k0/(QI8_0/2)],
-                    y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
+                    y_df[j*MMQ_TILE_Y_K_LDS + k01/QI8_1]);
             }
         }
     }
@@ -1279,7 +1369,7 @@ static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma(
     constexpr int rows_per_warp = granularity;
     constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
-    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
+    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K_LDS);
 
     const int   * x_qs = (const int   *) x;
     const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
@@ -1300,10 +1390,10 @@ static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma(
 #pragma unroll
         for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
             tile_B B[1];
-            load_generic(((tile_load *) B)[0], y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
+            load_generic(((tile_load *) B)[0], y_qs + j0*MMQ_TILE_Y_K_LDS + k01, MMQ_TILE_Y_K_LDS);
 
             const int j = j0 + tile_C::get_j(0);
-            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1] / 2;
+            const float dB = y_df[j*MMQ_TILE_Y_K_LDS + k01/QI8_1] / 2;
 
 #pragma unroll
             for (int n = 0; n < ntx; ++n) {
@@ -1328,7 +1418,7 @@ static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma(
     constexpr int rows_per_warp = granularity;
     constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
-    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
+    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K_LDS);
 
     const int   * x_qs = (const int   *) x;
     const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
@@ -1349,10 +1439,10 @@ static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma(
 #pragma unroll
         for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
             tile_B B;
-            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
+            load_generic(B, y_qs + j0*MMQ_TILE_Y_K_LDS + k01, MMQ_TILE_Y_K_LDS);
 
             const int j = j0 + tile_C::get_j(0);
-            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
+            const float dB = y_df[j*MMQ_TILE_Y_K_LDS + k01/QI8_1];
 
 #pragma unroll
             for (int n = 0; n < ntx; ++n) {
@@ -1378,7 +1468,7 @@ static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma(
     constexpr int rows_per_warp = 2 * granularity;
     constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
-    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
+    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K_LDS);
 
     const int   * x_qs = (const int   *) x;
     const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
@@ -1420,14 +1510,14 @@ static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma(
             float dB[tile_C::ne/2];
 
             // Here load_generic is faster than load_ldmatrix.
-            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
-            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);
+            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K_LDS + (k01 + 0),         MMQ_TILE_Y_K_LDS);
+            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K_LDS + (k01 + tile_B::J), MMQ_TILE_Y_K_LDS);
 
 #pragma unroll
             for (int l = 0; l < tile_C::ne/2; ++l) {
                 const int j = j0 + tile_C::get_j(l);
 
-                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
+                dB[l] = y_df[j*MMQ_TILE_Y_K_LDS + k01/QI8_1];
             }
 
 #pragma unroll
@@ -1524,7 +1614,7 @@ static __device__ __forceinline__ void vec_dot_q2_K_q8_1_dp4a(
     for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
         const int j = j0 + threadIdx.y;
 
-        y_df[j0/nwarps] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
+        y_df[j0/nwarps] = __half22float2(y_ds[j*MMQ_TILE_Y_K_LDS]);
     }
 
 #pragma unroll
@@ -1541,9 +1631,9 @@ static __device__ __forceinline__ void vec_dot_q2_K_q8_1_dp4a(
 
                 constexpr int ns = 2;
                 sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
-                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
+                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K_LDS + k01],
                     &x_dm[i*(MMQ_TILE_NE_K + 1) + k0/4], k01 < MMQ_TILE_NE_K/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
-                    &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
+                    &y_ds[j*MMQ_TILE_Y_K_LDS + (1 + k01/QI8_1)]);
             }
         }
     }
@@ -1564,9 +1654,9 @@ static __device__ __forceinline__ void vec_dot_q2_K_q8_1_dp4a(
 
                 constexpr int ns = 1;
                 sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
-                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
+                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K_LDS + k01],
                     &x_dm[i*(MMQ_TILE_NE_K + 1) + k0/4], k01 < MMQ_TILE_NE_K/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
-                    &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
+                    &y_ds[j*MMQ_TILE_Y_K_LDS + (1 + k01/QI8_1)]);
             }
         }
     }
@@ -1586,7 +1676,7 @@ static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
     constexpr int rows_per_warp = granularity;
     constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
-    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
+    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K_LDS);
 
     const int   * x_qs = (const int   *) x;
     const half2 * x_dm = (const half2 *) x_qs + MMQ_TILE_NE_K*2;
@@ -1607,13 +1697,13 @@ static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
 #pragma unroll
         for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
             tile_B B[1];
-            load_generic(((tile_load *) B)[0], y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
+            load_generic(((tile_load *) B)[0], y_qs + j0*MMQ_TILE_Y_K_LDS + k01, MMQ_TILE_Y_K_LDS);
 
             const int j = j0 + tile_C::get_j(0);
-            const float dB = (k01 < MMQ_TILE_NE_K/2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K]).x/2 : __half22float2(y_ds[j*MMQ_TILE_Y_K]).y/2;
+            const float dB = (k01 < MMQ_TILE_NE_K/2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K_LDS]).x/2 : __half22float2(y_ds[j*MMQ_TILE_Y_K_LDS]).y/2;
             const float sB = (k01 >= MMQ_TILE_NE_K * 3/4) ? 0
-                                              : (((k01/4)%2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]).y
-                                                             : __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]).x);
+                                              : (((k01/4)%2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K_LDS + (1 + k01/QI8_1)]).y
+                                                             : __half22float2(y_ds[j*MMQ_TILE_Y_K_LDS + (1 + k01/QI8_1)]).x);
 
             tile_C Cm;
             if (k01 >= MMQ_TILE_NE_K * 3/4) {
@@ -1652,7 +1742,7 @@ static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
     constexpr int rows_per_warp = granularity;
     constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
-    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
+    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K_LDS);
 
     const int   * x_qs = (const int   *) x;
     const half2 * x_dm = (const half2 *) x_qs + MMQ_TILE_NE_K*2;
@@ -1673,13 +1763,13 @@ static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
 #pragma unroll
         for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
             tile_B B;
-            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
+            load_generic(B, y_qs + j0*MMQ_TILE_Y_K_LDS + k01, MMQ_TILE_Y_K_LDS);
 
             const int j = j0 + tile_C::get_j(0);
-            const float dB = (k01 < MMQ_TILE_NE_K/2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K]).x : __half22float2(y_ds[j*MMQ_TILE_Y_K]).y;
+            const float dB = (k01 < MMQ_TILE_NE_K/2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K_LDS]).x : __half22float2(y_ds[j*MMQ_TILE_Y_K_LDS]).y;
             const float sB = (k01 >= MMQ_TILE_NE_K * 3/4) ? 0
-                                              : (((k01/4)%2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]).y
-                                                             : __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]).x);
+                                              : (((k01/4)%2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K_LDS + (1 + k01/QI8_1)]).y
+                                                             : __half22float2(y_ds[j*MMQ_TILE_Y_K_LDS + (1 + k01/QI8_1)]).x);
 
             tile_C Cm;
             if (k01 >= MMQ_TILE_NE_K * 3/4) {
@@ -1721,7 +1811,7 @@ static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
     constexpr int rows_per_warp = 2 * granularity;
     constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
-    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
+    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K_LDS);
 
     const int   * x_qs = (const int   *) x;
     const half2 * x_dm = (const half2 *) x_qs + MMQ_TILE_NE_K*2;
@@ -1770,7 +1860,7 @@ static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
         for (int l = 0; l < tile_C::ne/2; ++l) {
             const int j = j0 + tile_C::get_j(l);
 
-            dB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
+            dB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K_LDS]);
         }
 
 #pragma unroll
@@ -1778,8 +1868,8 @@ static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
             tile_B B[2];
 
             // Here load_generic is faster than load_ldmatrix.
-            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
-            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);
+            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K_LDS + (k01 + 0),         MMQ_TILE_Y_K_LDS);
+            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K_LDS + (k01 + tile_B::J), MMQ_TILE_Y_K_LDS);
 
             tile_C Cm[2];
             if (k01 >= MMQ_TILE_NE_K * 3/4) {
@@ -1816,7 +1906,7 @@ static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
             for (int l = 0; l < tile_C::ne/2; ++l) {
                 const int j = j0 + tile_C::get_j(l);
 
-                sB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
+                sB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K_LDS + (1 + k01/QI8_1)]);
             }
 
 #pragma unroll
@@ -1964,8 +2054,8 @@ static __device__ __forceinline__ void vec_dot_q3_K_q8_1_dp4a(
                 const int8_t * scales = ((const int8_t *) (x_sc + i*(MMQ_TILE_NE_K/8) + i/8)) + k0/4;
 
                 sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q3_K_q8_1_impl_mmq(
-                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], scales,
-                    x_df[i], y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
+                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K_LDS + k01], scales,
+                    x_df[i], y_df[j*MMQ_TILE_Y_K_LDS + k01/QI8_1]);
             }
         }
     }
@@ -2118,8 +2208,8 @@ static __device__ __forceinline__ void vec_dot_q4_K_q8_1_dp4a(
                 const uint8_t * sc = (const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k0/32] + 2*(k01/16);
 
                 sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q4_K_q8_1_impl_mmq(
-                    &x_qs[i*(MMQ_TILE_NE_K + 1) + k0/2], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
-                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
+                    &x_qs[i*(MMQ_TILE_NE_K + 1) + k0/2], &y_qs[j*MMQ_TILE_Y_K_LDS + k01], sc, sc+8,
+                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K_LDS + k01/QI8_1]);
             }
         }
     }
@@ -2275,8 +2365,8 @@ static __device__ __forceinline__ void vec_dot_q5_K_q8_1_dp4a(
                 const uint8_t * sc = ((const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k00/32]) + 2*(k01/16);
 
                 sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q5_K_q8_1_impl_mmq(
-                    &x_qs[i*(QR5_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
-                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
+                    &x_qs[i*(QR5_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K_LDS + k01], sc, sc+8,
+                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K_LDS + k01/QI8_1]);
             }
         }
     }
@@ -2396,8 +2486,8 @@ static __device__ __forceinline__ void vec_dot_q6_K_q8_1_dp4a(
                 const int8_t * sc = ((const int8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k0/16]);
 
                 sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q6_K_q8_1_impl_mmq(
-                    &x_qs[i*(QR6_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc,
-                    x_df[i*(MMQ_TILE_NE_K/QI6_K) + i/QI6_K], &y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
+                    &x_qs[i*(QR6_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K_LDS + k01], sc,
+                    x_df[i*(MMQ_TILE_NE_K/QI6_K) + i/QI6_K], &y_df[j*MMQ_TILE_Y_K_LDS + k01/QI8_1]);
             }
         }
     }
@@ -2417,7 +2507,7 @@ static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
     constexpr int rows_per_warp = granularity;
     constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
-    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
+    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K_LDS);
 
     const int   * x_qs = (const int   *) x;
     const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
@@ -2439,10 +2529,10 @@ static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
 #pragma unroll
         for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
             tile_B B[1];
-            load_generic(((tile_load *) B)[0], y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
+            load_generic(((tile_load *) B)[0], y_qs + j0*MMQ_TILE_Y_K_LDS + k01, MMQ_TILE_Y_K_LDS);
 
             const int j = j0 + tile_C::get_j(0);
-            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1] / 2;
+            const float dB = y_df[j*MMQ_TILE_Y_K_LDS + k01/QI8_1] / 2;
 
 #pragma unroll
             for (int n = 0; n < ntx; ++n) {
@@ -2468,7 +2558,7 @@ static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
     constexpr int rows_per_warp = granularity;
     constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
-    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
+    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K_LDS);
 
     const int   * x_qs = (const int   *) x;
     const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
@@ -2490,10 +2580,10 @@ static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
 #pragma unroll
         for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
             tile_B B;
-            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
+            load_generic(B, y_qs + j0*MMQ_TILE_Y_K_LDS + k01, MMQ_TILE_Y_K_LDS);
 
             const int j = j0 + tile_C::get_j(0);
-            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
+            const float dB = y_df[j*MMQ_TILE_Y_K_LDS + k01/QI8_1];
 
 #pragma unroll
             for (int n = 0; n < ntx; ++n) {
@@ -2519,7 +2609,7 @@ static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
     constexpr int rows_per_warp = 2 * granularity;
     constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
-    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
+    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K_LDS);
 
     const int   * x_qs = (const int   *) x;
     const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
@@ -2579,14 +2669,14 @@ static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
             float dB[tile_C::ne/2];
 
             // Here load_generic is faster than load_ldmatrix.
-            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + 0         + k01, MMQ_TILE_Y_K);
-            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + tile_B::J + k01, MMQ_TILE_Y_K);
+            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K_LDS + 0         + k01, MMQ_TILE_Y_K_LDS);
+            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K_LDS + tile_B::J + k01, MMQ_TILE_Y_K_LDS);
 
 #pragma unroll
             for (int l = 0; l < tile_C::ne/2; ++l) {
                 const int j = j0 + tile_C::get_j(l);
 
-                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
+                dB[l] = y_df[j*MMQ_TILE_Y_K_LDS + k01/QI8_1];
             }
 
 #pragma unroll
@@ -3375,7 +3465,7 @@ static __device__ __forceinline__ void mul_mat_q_process_tile(
 
     extern __shared__ int data_mul_mat_q[];
     int * tile_y = data_mul_mat_q + mmq_x;
-    int * tile_x = tile_y + GGML_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size);
+    int * tile_x = tile_y + GGML_PAD(mmq_x*MMQ_TILE_Y_K_LDS, nwarps*warp_size);
 
 #if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
     constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, need_check, type>::vec_dot_mma;
@@ -3404,15 +3494,22 @@ static __device__ __forceinline__ void mul_mat_q_process_tile(
         {
             const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
 #pragma unroll
-            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
-                int l = l0 + threadIdx.y*warp_size + threadIdx.x;
-
+            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*warp_size) {
+                const int l = l0 + threadIdx.y*warp_size + threadIdx.x;
                 tile_y[l] = by0[l];
             }
         }
 
         __syncthreads();
 
+// GFX906 PREFETCH: Issue AFTER barrier1, BEFORE vec_dot1
+// Maximum overlap: vec_dot1 + barrier2 + Y_tile2_load + barrier3 + vec_dot2 + barrier4 + X_tile_loads
+// Data used at Y_tile1_load in next iteration (~600+ instructions of overlap)
+#if defined(GGML_USE_HIP) && defined(__gfx906__)
+        int prefetch_keep_alive = gfx906_prefetch_y_tile_v4<mmq_x, MMQ_TILE_Y_K, nwarps, warp_size>(
+            y, ncols_y, kb0, kb0_stop, qk, blocks_per_iter);
+#endif
+
         vec_dot(tile_x, tile_y, sum, 0);
 
         __syncthreads();
@@ -3420,9 +3517,8 @@ static __device__ __forceinline__ void mul_mat_q_process_tile(
         {
             const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
 #pragma unroll
-            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
-                int l = l0 + threadIdx.y*warp_size + threadIdx.x;
-
+            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*warp_size) {
+                const int l = l0 + threadIdx.y*warp_size + threadIdx.x;
                 tile_y[l] = by0[l];
             }
         }
@@ -3431,6 +3527,10 @@ static __device__ __forceinline__ void mul_mat_q_process_tile(
 
         vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);
 
+#if defined(GGML_USE_HIP) && defined(__gfx906__)
+        gfx906_prefetch_consume(prefetch_keep_alive);
+#endif
+
         __syncthreads();
     }
 
diff --git a/ggml/src/ggml-cuda/mmvq.cu b/ggml/src/ggml-cuda/mmvq.cu
index d671551c1..32dbbfba1 100644
--- a/ggml/src/ggml-cuda/mmvq.cu
+++ b/ggml/src/ggml-cuda/mmvq.cu
@@ -3,6 +3,13 @@
 #include "unary.cuh"
 #include "vecdotq.cuh"
 
+// GFX906-specific warp-cooperative MMVQ kernels (compile with -DGGML_HIP_GFX906)
+#if defined(GGML_HIP_GFX906)
+#include "gfx906/gfx906-mmvq-q4_0.cuh"
+#include "gfx906/gfx906-mmvq-q4_1.cuh"
+#include "gfx906/gfx906-mmvq-q8_0.cuh"
+#endif
+
 #include <cstdint>
 
 typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);
@@ -477,12 +484,53 @@ static void mul_mat_vec_q_switch_type(
         cudaStream_t stream) {
     switch (type_x) {
         case GGML_TYPE_Q4_0:
+#if defined(GGML_HIP_GFX906)
+            // GFX906: Use warp-cooperative kernel for ncols_dst=1 (token generation) without fusion
+            {
+                const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
+
+                if (ncols_dst == 1 && !has_fusion && ncols_x <= 1024) {
+                    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
+                    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
+                    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);
+
+                    gfx906_launch_mul_mat_vec_q4_0_warp_coop(
+                        vx, vy, ids, dst,
+                        ncols_x, nchannels_y_fd, stride_row_x, stride_col_dst,
+                        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
+                        sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
+                        nrows_x, nchannels_dst, nsamples_dst, stream);
+                    break;
+                }
+            }
+#endif
             mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                 (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                  nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                  nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
             break;
         case GGML_TYPE_Q4_1:
+#if defined(GGML_HIP_GFX906)
+            // GFX906: Use warp-cooperative kernel for ncols_dst=1 (token generation) without fusion
+            {
+                const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
+
+                // Use warp-coop for small matrices only (MoE experts, <= 1024 cols)
+                if (ncols_dst == 1 && !has_fusion && ncols_x <= 1024) {
+                    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
+                    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
+                    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);
+
+                    gfx906_launch_mul_mat_vec_q4_1_warp_coop(
+                        vx, vy, ids, dst,
+                        ncols_x, nchannels_y_fd, stride_row_x, stride_col_dst,
+                        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
+                        sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
+                        nrows_x, nchannels_dst, nsamples_dst, stream);
+                    break;
+                }
+            }
+#endif
             mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                 (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                  nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
@@ -501,6 +549,26 @@ static void mul_mat_vec_q_switch_type(
                  nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
             break;
         case GGML_TYPE_Q8_0:
+#if defined(GGML_HIP_GFX906)
+            // GFX906: Use warp-cooperative kernel for ncols_dst=1 (token generation) without fusion
+            {
+                const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
+
+                if (ncols_dst == 1 && !has_fusion && ncols_x <= 1024) {
+                    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
+                    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
+                    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);
+
+                    gfx906_launch_mul_mat_vec_q8_0_warp_coop(
+                        vx, vy, ids, dst,
+                        ncols_x, nchannels_y_fd, stride_row_x, stride_col_dst,
+                        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
+                        sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
+                        nrows_x, nchannels_dst, nsamples_dst, stream);
+                    break;
+                }
+            }
+#endif
             mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                 (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                  nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
diff --git a/ggml/src/ggml-cuda/quantize.cu b/ggml/src/ggml-cuda/quantize.cu
index a8c68e44b..d429ba40e 100644
--- a/ggml/src/ggml-cuda/quantize.cu
+++ b/ggml/src/ggml-cuda/quantize.cu
@@ -1,5 +1,7 @@
 #include "quantize.cuh"
 #include <cstdint>
+// #include <vector>
+// #include <cstdio>
 
 __launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
 static __global__ void quantize_q8_1(
@@ -293,6 +295,42 @@ void quantize_mmq_q8_1_cuda(
     GGML_ASSERT(ne00 % 4 == 0);
     GGML_ASSERT(ne0 % (4*QK8_1) == 0);
 
+    // --- HOST-SIDE DEBUG VALIDATION (commented out) ---
+    // fprintf(stderr, "[quantize_mmq_q8_1] ne00=%ld s01=%ld s02=%ld s03=%ld ne0=%ld ne1=%ld ne2=%ld ne3=%ld ids=%p x=%p\n",
+    //         (long)ne00, (long)s01, (long)s02, (long)s03, (long)ne0, (long)ne1, (long)ne2, (long)ne3, (void*)ids, (void*)x);
+    //
+    // // Calculate maximum index that will be accessed
+    // int64_t max_i1_val = ne1 - 1;
+    // int64_t max_src_idx = (ne3-1)*s03 + (ne2-1)*s02 + max_i1_val*s01 + (ne0-4);
+    // fprintf(stderr, "[quantize_mmq_q8_1] Grid: (%ld, %ld, %ld) max_src_idx=%ld (assuming no ids remapping)\n",
+    //         (long)ne1, (long)((ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ)),
+    //         (long)(ne2*ne3), (long)max_src_idx);
+    //
+    // if (ids) {
+    //     // For MoE: copy ids to host and validate
+    //     std::vector<int32_t> ids_host(ne1);
+    //     CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids, ne1 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
+    //     CUDA_CHECK(cudaStreamSynchronize(stream));
+    //
+    //     int64_t max_ids_val = 0;
+    //     int64_t min_ids_val = INT64_MAX;
+    //     for (int64_t i = 0; i < ne1; i++) {
+    //         if (ids_host[i] > max_ids_val) max_ids_val = ids_host[i];
+    //         if (ids_host[i] < min_ids_val) min_ids_val = ids_host[i];
+    //     }
+    //     fprintf(stderr, "[quantize_mmq_q8_1] ids range: [%ld, %ld] (ne1=%ld)\n",
+    //             (long)min_ids_val, (long)max_ids_val, (long)ne1);
+    //
+    //     max_src_idx = (ne3-1)*s03 + (ne2-1)*s02 + max_ids_val*s01 + (ne0-4);
+    //     fprintf(stderr, "[quantize_mmq_q8_1] max_src_idx with ids remapping=%ld\n", (long)max_src_idx);
+    //
+    //     if (max_ids_val * s01 > 1000000000LL) {
+    //         fprintf(stderr, "[quantize_mmq_q8_1] WARNING: max_ids_val * s01 = %ld seems too large!\n",
+    //                 (long)(max_ids_val * s01));
+    //     }
+    // }
+    // --- END DEBUG VALIDATION ---
+
     // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
     const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
     const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
diff --git a/ggml/src/ggml-cuda/softmax.cu b/ggml/src/ggml-cuda/softmax.cu
index 1ae84ebf6..eeacde0bd 100644
--- a/ggml/src/ggml-cuda/softmax.cu
+++ b/ggml/src/ggml-cuda/softmax.cu
@@ -1,14 +1,6 @@
 #include "common.cuh"
 #include "ggml.h"
 #include "softmax.cuh"
-
-#ifdef GGML_USE_HIP
-#include <hip/hip_cooperative_groups.h>
-#else
-#include <cooperative_groups.h>
-#include <cooperative_groups/reduce.h>
-#endif // GGML_USE_HIP
-
 #include <cstdint>
 #include <utility>
 
@@ -168,156 +160,6 @@ static __global__ void soft_max_f32(
         dst[col] = vals[col] * inv_sum;
     }
 }
-
-
-// TODO: This is a common pattern used across kernels that could be moved to common.cuh + templated
-static __device__ float two_stage_warp_reduce_max(float val) {
-    val = warp_reduce_max(val);
-    if (blockDim.x > WARP_SIZE) {
-        assert((blockDim.x <= 1024) && (blockDim.x % WARP_SIZE) == 0);
-        __shared__ float local_vals[32];
-        const int        warp_id = threadIdx.x / WARP_SIZE;
-        const int        lane_id = threadIdx.x % WARP_SIZE;
-        if (lane_id == 0) {
-            local_vals[warp_id] = val;
-        }
-        __syncthreads();
-        val = -INFINITY;
-        if (lane_id < (static_cast<int>(blockDim.x) / WARP_SIZE)) {
-            val = local_vals[lane_id];
-        }
-        return warp_reduce_max(val);
-    } else {
-        return val;
-    }
-}
-
-static __device__ float two_stage_warp_reduce_sum(float val) {
-    val = warp_reduce_sum(val);
-    if (blockDim.x > WARP_SIZE) {
-        assert((blockDim.x <= 1024) && (blockDim.x % WARP_SIZE) == 0);
-        __shared__ float local_vals[32];
-        const int        warp_id = threadIdx.x / WARP_SIZE;
-        const int        lane_id = threadIdx.x % WARP_SIZE;
-        if (lane_id == 0) {
-            local_vals[warp_id] = val;
-        }
-        __syncthreads();
-        val = 0.0f;
-        if (lane_id < (static_cast<int>(blockDim.x) / WARP_SIZE)) {
-            val = local_vals[lane_id];
-        }
-        return warp_reduce_sum(val);
-    } else {
-        return val;
-    }
-}
-
-// TODO: Template to allow keeping ncols in registers if they fit
-static __device__ void soft_max_f32_parallelize_cols_single_row(const float * __restrict__ x,
-                                                                float * __restrict__ dst,
-                                                                float * __restrict__ tmp_maxs,
-                                                                float * __restrict__ tmp_sums,
-                                                                const soft_max_params p) {
-    namespace cg = cooperative_groups;
-
-    const cg::grid_group g = cg::this_grid();
-
-    const int tid               = threadIdx.x;
-    const int col_start         = blockIdx.x * blockDim.x + tid;
-    const int n_elem_per_thread = 4;
-
-    float     local_vals[n_elem_per_thread] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
-    float     local_max                     = -INFINITY;
-    const int step_size                     = gridDim.x * blockDim.x;
-
-    // Compute thread-local max
-    for (int col = col_start; col < p.ncols;) {
-#pragma unroll
-        for (int i = 0; i < n_elem_per_thread; i++) {
-            const int idx = col + i * step_size;
-            local_vals[i] = idx < p.ncols ? x[idx] : -INFINITY;
-        }
-#pragma unroll
-        for (int i = 0; i < n_elem_per_thread; i++) {
-            local_max = fmaxf(local_max, local_vals[i]);
-        }
-        col += step_size * n_elem_per_thread;
-    }
-
-    // Compute CTA-level max
-    local_max = two_stage_warp_reduce_max(local_max);
-
-    // Store CTA-level max to GMEM
-    if (tid == 0) {
-        tmp_maxs[blockIdx.x] = local_max;
-    }
-    g.sync();
-
-    // Compute compute global max from CTA-level maxs
-    assert(gridDim.x < blockDim.x);  // currently we only support this case
-    if (tid < gridDim.x) {
-        local_max = tmp_maxs[tid];
-    } else {
-        local_max = -INFINITY;
-    }
-    local_max = two_stage_warp_reduce_max(local_max);
-
-    // Compute softmax dividends, accumulate divisor
-    float tmp_expf = 0.0f;
-    for (int col = col_start; col < p.ncols;) {
-#pragma unroll
-        for (int i = 0; i < n_elem_per_thread; i++) {
-            const int idx = col + i * step_size;
-            local_vals[i] = idx < p.ncols ? x[idx] : -INFINITY;
-        }
-#pragma unroll
-        for (int i = 0; i < n_elem_per_thread; i++) {
-            const int idx = col + i * step_size;
-            if (idx < p.ncols) {
-                const float tmp = expf(local_vals[i] - local_max);
-                tmp_expf += tmp;
-                dst[idx] = tmp;
-            }
-        }
-        col += step_size * n_elem_per_thread;
-    }
-
-    // Reduce divisor within CTA
-    tmp_expf = two_stage_warp_reduce_sum(tmp_expf);
-
-    // Store CTA-level sum to GMEM
-    if (tid == 0) {
-        tmp_sums[blockIdx.x] = tmp_expf;
-    }
-    g.sync();
-
-    // Compute global sum from CTA-level sums
-    if (tid < gridDim.x) {
-        tmp_expf = tmp_sums[tid];
-    } else {
-        tmp_expf = 0.0f;
-    }
-    tmp_expf = two_stage_warp_reduce_sum(tmp_expf);
-
-    // Divide dividend by global sum + store data
-    for (int col = col_start; col < p.ncols;) {
-#pragma unroll
-        for (int i = 0; i < n_elem_per_thread; i++) {
-            const int idx = col + i * step_size;
-            local_vals[i] = idx < p.ncols ? dst[idx] : -INFINITY;
-        }
-#pragma unroll
-        for (int i = 0; i < n_elem_per_thread; i++) {
-            const int idx = col + i * step_size;
-            if (idx < p.ncols) {
-                dst[idx] = local_vals[i] / tmp_expf;
-            }
-        }
-        col += step_size * n_elem_per_thread;
-    }
-}
-
 #ifdef __clang__
 #pragma clang diagnostic pop
 #endif // __clang__
@@ -374,31 +216,9 @@ static void launch_soft_max_kernels(const float * x, const T * mask, const float
     soft_max_f32<true, 0, 0><<<block_nums, block_dims, nbytes_shared, stream>>>(x, mask, sinks, dst, p);
 }
 
-__launch_bounds__(8*WARP_SIZE, 1) static __global__ void soft_max_f32_parallelize_cols(const float * __restrict__ x,
-                                                     float * __restrict__ dst,
-                                                     float * __restrict__ tmp_maxs,
-                                                     float * __restrict__ tmp_sums,
-                                                     const soft_max_params p)
-// We loop over all instead of parallelizing across gridDim.y as cooperative groups
-// currently only support synchronizing the complete grid if not launched as a cluster group
-// (which requires CC > 9.0)
-// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#grid-synchronization
-// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#class-cluster-group
-{
-    for (int rowx = 0; rowx < p.ne01 * p.ne02 * p.ne03; rowx++) {
-        soft_max_f32_parallelize_cols_single_row(x + int64_t(rowx) * p.ncols, dst + int64_t(rowx) * p.ncols, tmp_maxs,
-                                                 tmp_sums, p);
-    }
-}
 
-template <typename T>
-static void soft_max_f32_cuda(const float *                                x,
-                              const T *                                    mask,
-                              const float *                                sinks,
-                              float *                                      dst,
-                              const soft_max_params &                      params,
-                              cudaStream_t                                 stream,
-                              [[maybe_unused]] ggml_backend_cuda_context & ctx) {
+template<typename T>
+static void soft_max_f32_cuda(const float * x, const T * mask, const float * sinks, float * dst, const soft_max_params & params, cudaStream_t stream) {
     int nth = WARP_SIZE;
     const int64_t ncols_x = params.ncols;
 
@@ -416,25 +236,8 @@ static void soft_max_f32_cuda(const float *                                x,
     if (nbytes_shared <= smpbo) {
         launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(x, mask, sinks, dst, params, stream, block_dims, block_nums, nbytes_shared);
     } else {
-        // Parallelize across SMs for top-p/dist-sampling
-        // The heuristic for parallelizing rows across SMs vs parallelizing single row & looping over all rows was done on the basis of a B6000 GPU and
-        // Can be adapted further for lower-SM-count GPUs, though keeping data in registers should be implemented first as that is the optimal solution.
-        if (ggml_cuda_info().devices[id].supports_cooperative_launch &&
-            ncols_x / (params.ne01 * params.ne02 * params.ne03) > 8192 && mask == nullptr && sinks == nullptr &&
-            params.scale == 1.0f && params.max_bias == 0.0f) {
-            ggml_cuda_pool_alloc<float> tmp_maxs_alloc(ctx.pool(), ggml_cuda_info().devices[id].nsm * sizeof(float));
-            ggml_cuda_pool_alloc<float> tmp_sums_alloc(ctx.pool(), ggml_cuda_info().devices[id].nsm * sizeof(float));
-
-            void * kernel_args[] = { (void *) &x, (void *) &dst, (void *) &tmp_maxs_alloc.ptr,
-                                     (void *) &tmp_sums_alloc.ptr, (void *) const_cast<soft_max_params *>(&params) };
-            CUDA_CHECK(cudaLaunchCooperativeKernel((void *) soft_max_f32_parallelize_cols,
-                                                   dim3(ggml_cuda_info().devices[id].nsm, 1, 1),
-                                                   dim3(WARP_SIZE * 8, 1, 1), kernel_args, 0, stream));
-        } else {
-            const size_t nbytes_shared_low = WARP_SIZE * sizeof(float);
-            soft_max_f32<false, 0, 0>
-                <<<block_nums, block_dims, nbytes_shared_low, stream>>>(x, mask, sinks, dst, params);
-        }
+        const size_t nbytes_shared_low = WARP_SIZE*sizeof(float);
+        soft_max_f32<false, 0, 0><<<block_nums, block_dims, nbytes_shared_low, stream>>>(x, mask, sinks, dst, params);
     }
 }
 
@@ -512,9 +315,9 @@ void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     params.m1 = m1;
 
     if (use_f16) {
-        soft_max_f32_cuda(src0_d, (const half *) src1_d, (const float *) src2_d, dst_d, params, stream, ctx);
+        soft_max_f32_cuda(src0_d, (const half  *) src1_d, (const float *) src2_d, dst_d, params, stream);
     } else {
-        soft_max_f32_cuda(src0_d, (const float *) src1_d, (const float *) src2_d, dst_d, params, stream, ctx);
+        soft_max_f32_cuda(src0_d, (const float *) src1_d, (const float *) src2_d, dst_d, params, stream);
     }
 }
 
diff --git a/ggml/src/ggml-cuda/ssm-scan.cu b/ggml/src/ggml-cuda/ssm-scan.cu
index c1d4e2bc8..6b424381d 100644
--- a/ggml/src/ggml-cuda/ssm-scan.cu
+++ b/ggml/src/ggml-cuda/ssm-scan.cu
@@ -114,7 +114,7 @@ __global__ void __launch_bounds__(splitD, 1)
 #endif // __clang__
 
 // assumes as many threads as d_state
-template <int c_factor, int d_state>
+template <int splitH, int d_state>
 __global__ void __launch_bounds__(d_state, 1)
     ssm_scan_f32_group(
         const float * __restrict__ src0, const float * __restrict__ src1, const float * __restrict__ src2,
@@ -125,25 +125,20 @@ __global__ void __launch_bounds__(d_state, 1)
         const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
         const int64_t s_off, const int64_t n_head, const int64_t d_head, const int64_t n_group, const int64_t n_tok) {
 
-    const int warp     = threadIdx.x / WARP_SIZE;
-    const int lane     = threadIdx.x % WARP_SIZE;
-    const int warp_idx = blockIdx.x  * c_factor + warp;
-
-    const int head_idx =  warp_idx / d_head;
-    const int head_off = (warp_idx % d_head) * sizeof(float);
-    const int seq_idx  = blockIdx.y;
+    const int head_idx = (blockIdx.x * splitH) / d_head;
+    const int head_off = ((blockIdx.x * splitH) % d_head) * sizeof(float);
+    const int seq_idx = blockIdx.y;
 
     const int group_off = (head_idx / (n_head / n_group)) * d_state * sizeof(float);
 
-    // TODO: refactor strides to be in elements/floats instead of bytes to be cleaner and consistent with the rest of the codebase
-    const float * s0_warp = (const float *) ((const char *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
-    const float * x_warp  = (const float *) ((const char *) src1 + (seq_idx * src1_nb3) + (warp_idx * sizeof(float)));
-    const float * dt_warp = (const float *) ((const char *) src2 + (seq_idx * src2_nb2) + head_idx * sizeof(float));
-    const float * A_warp  = (const float *) ((const char *) src3 + head_idx * src3_nb1);
-    const float * B_warp  = (const float *) ((const char *) src4 + (seq_idx * src4_nb3) + (group_off));
-    const float * C_warp  = (const float *) ((const char *) src5 + (seq_idx * src5_nb3) + (group_off));
-    float *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
-    float *       s_warp  = (float *) ((char *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
+    const float * s0_block = (const float *) ((const char *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
+    const float * x_block  = (const float *) ((const char *) src1 + (seq_idx * src1_nb3) + blockIdx.x * splitH * sizeof(float));
+    const float * dt_block = (const float *) ((const char *) src2 + (seq_idx * src2_nb2) + head_idx * sizeof(float));
+    const float * A_block  = (const float *) ((const char *) src3 + head_idx * src3_nb1);
+    const float * B_block  = (const float *) ((const char *) src4 + (seq_idx * src4_nb3) + (group_off));
+    const float * C_block  = (const float *) ((const char *) src5 + (seq_idx * src5_nb3) + (group_off));
+    float *       y_block  = dst + (seq_idx * n_tok * n_head * d_head) + blockIdx.x * splitH;
+    float *       s_block  = (float *) ((char *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
 
     // strides across n_seq_tokens
     const int stride_x  = src1_nb2 / sizeof(float);
@@ -152,42 +147,80 @@ __global__ void __launch_bounds__(d_state, 1)
     const int stride_C  = src5_nb2 / sizeof(float);
     const int stride_y  = n_head * d_head;
 
-    float state[c_factor];
-    float state_sum = 0.0f;
+    float state[splitH];
+    // for the parallel accumulation
+    __shared__ float stateC[splitH * d_state];
 
 #pragma unroll
-    for (int j = 0; j < c_factor; j++) {
-        state[j] = s0_warp[WARP_SIZE * j + lane];
+    for (int j = 0; j < splitH; j++) {
+        state[j] = s0_block[j * d_state + threadIdx.x];
     }
 
     for (int64_t i = 0; i < n_tok; i++) {
-        // NOTE: dt_soft_plus, dA and x_dt have the same value for a warp here.
-        // Recalculation is intentional; sharing via shuffles/smem proved slower due to sync overhead.
-        const float dt_soft_plus = (dt_warp[i * stride_dt] <= 20.0f ? log1pf(expf(dt_warp[i * stride_dt])) : dt_warp[i * stride_dt]);
+        // TODO: only calculate dA and dt_soft_plus once per head instead of every splitH head elements
+        // TODO: only calculate B and C once per head group
+        // NOTE: dt_soft_plus, dA and x_dt have the same value across threads here.
+        float dt_soft_plus = dt_block[i * stride_dt];
+        if (dt_soft_plus <= 20.0f) {
+            dt_soft_plus = log1pf(expf(dt_soft_plus));
+        }
+        const float dA = expf(dt_soft_plus * A_block[0]);
+        const float B = B_block[i * stride_B + threadIdx.x];
+        const float C = C_block[i * stride_C + threadIdx.x];
 
-        state_sum = 0.0f;
-        const float dA   = expf(dt_soft_plus * A_warp[0]);
-        const float x_dt = x_warp[i * stride_x] * dt_soft_plus;
+        // across d_head
 #pragma unroll
-        for (int j = 0; j < c_factor; j++) {
-            const float B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
-            const float C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
-            state[j] = (state[j] * dA) + (B_val * x_dt);
-            state_sum += state[j] * C_val;
+        for (int j = 0; j < splitH; j++) {
+            const float x_dt = x_block[i * stride_x + j] * dt_soft_plus;
+
+            state[j] = (state[j] * dA) + (B * x_dt);
+
+            stateC[j * d_state + threadIdx.x] = state[j] * C;
         }
 
-        // parallel accumulation for output
-        state_sum = warp_reduce_sum(state_sum);
+        __syncthreads();
+
+        // parallel accumulation for stateC
+        // TODO: simplify
+        {
+            static_assert((d_state & -d_state) == d_state, "the state size has to be a power of 2");
+            static_assert((splitH & -splitH) == splitH, "splitH has to be a power of 2");
+
+            // reduce until w matches the warp size
+            // TODO: does this work even when the physical warp size is 64?
+#pragma unroll
+            for (int w = d_state; w > WARP_SIZE; w >>= 1) {
+                // (assuming there are d_state threads)
+#pragma unroll
+                for (int j = 0; j < ((w >> 1) * splitH + d_state - 1) / d_state; j++) {
+                    // TODO: check for bank conflicts
+                    const int k = (threadIdx.x % (w >> 1)) + (d_state * (threadIdx.x / (w >> 1))) + j * d_state * (d_state / (w >> 1));
+                    stateC[k] += stateC[k + (w >> 1)];
+
+                }
+                __syncthreads();
+            }
+
+            static_assert(splitH >= d_state / WARP_SIZE);
 
-        if (lane == 0) {
-            y_warp[i * stride_y] = state_sum;
+#pragma unroll
+            for (int j = 0; j < splitH / (d_state / WARP_SIZE); j++) {
+                float y = stateC[(threadIdx.x % WARP_SIZE) + d_state * (threadIdx.x / WARP_SIZE) + j * d_state * (d_state / WARP_SIZE)];
+                y = warp_reduce_sum(y);
+
+                // store the above accumulations
+                if (threadIdx.x % WARP_SIZE == 0) {
+                    const int k = threadIdx.x / WARP_SIZE + j * (d_state / WARP_SIZE);
+                    y_block[i * stride_y + k] = y;
+                }
+            }
         }
     }
 
     // write back the state
 #pragma unroll
-    for (int j = 0; j < c_factor; j++) {
-        s_warp[WARP_SIZE * j + lane] = state[j];
+    for (int j = 0; j < splitH; j++) {
+        s_block[j * d_state + threadIdx.x] = state[j];
     }
 }
 
@@ -198,24 +231,27 @@ static void ssm_scan_f32_cuda(const float * src0, const float * src1, const floa
                               const int src5_nb3, const int64_t s_off, const int64_t d_state, const int64_t head_dim,
                               const int64_t n_head, const int64_t n_group, const int64_t n_tok, const int64_t n_seq,
                               cudaStream_t stream) {
+    const int threads = 128;
     // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
     if (src3_nb1 == sizeof(float)) {
         // Mamba-2
         if (d_state == 128) {
-            constexpr int threads   = 128;
-            constexpr int num_warps = threads/WARP_SIZE;
-
-            const dim3 blocks((n_head * head_dim + (num_warps - 1)) / num_warps, n_seq, 1);
-            ssm_scan_f32_group<128/WARP_SIZE, 128><<<blocks, threads, 0, stream>>>(
+            GGML_ASSERT(d_state % threads == 0);
+            // NOTE: can be any power of two between 4 and 64
+            const int splitH = 16;
+            GGML_ASSERT(head_dim % splitH == 0);
+            const dim3 blocks((n_head * head_dim + (splitH - 1)) / splitH, n_seq, 1);
+            ssm_scan_f32_group<16, 128><<<blocks, threads, 0, stream>>>(
                     src0, src1, src2, src3, src4, src5, src6, dst,
                     src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                     src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok);
         } else if (d_state == 256) { // Falcon-H1
-            constexpr int threads   = 256;
-            constexpr int num_warps = threads/WARP_SIZE;
-
-            const dim3 blocks((n_head * head_dim + (num_warps - 1)) / num_warps, n_seq, 1);
-            ssm_scan_f32_group<256/WARP_SIZE, 256><<<blocks, threads, 0, stream>>>(
+            const int threads = 256;
+            // NOTE: can be any power of two between 8 and 64
+            const int splitH = 16;
+            GGML_ASSERT(head_dim % splitH == 0);
+            const dim3 blocks((n_head * head_dim + (splitH - 1)) / splitH, n_seq, 1);
+            ssm_scan_f32_group<16, 256><<<blocks, threads, 0, stream>>>(
                     src0, src1, src2, src3, src4, src5, src6, dst,
                     src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                     src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok);
@@ -224,7 +260,6 @@ static void ssm_scan_f32_cuda(const float * src0, const float * src1, const floa
         }
     } else {
         // Mamba-1
-        constexpr int threads = 128;
         GGML_ASSERT(n_head % threads == 0);
         GGML_ASSERT(head_dim == 1);
         GGML_ASSERT(n_group == 1);
diff --git a/ggml/src/ggml-cuda/vecdotq.cuh b/ggml/src/ggml-cuda/vecdotq.cuh
index 6baab1176..031a797e9 100644
--- a/ggml/src/ggml-cuda/vecdotq.cuh
+++ b/ggml/src/ggml-cuda/vecdotq.cuh
@@ -4,6 +4,11 @@
 
 #include <cstdint>
 
+// GFX906 optimizations
+#if defined(GGML_USE_HIP) && defined(__gfx906__)
+    #include "gfx906/gfx906-vecdotq.cuh"
+#endif
+
 static __device__ __forceinline__ int get_int_b1(const void * x, const int & i32) {
     const uint8_t * x8 = (const uint8_t *) x;
 
@@ -15,8 +20,10 @@ static __device__ __forceinline__ int get_int_b1(const void * x, const int & i32
     return x32;
 }
 
+// GFX906: get_int_b1_fast is defined in gfx906/gfx906-vecdotq.cuh as gfx906_get_int_b1_fast
+
 static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
-    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment
+    const uint16_t * x16 = (const uint16_t *) x;
 
     int x32  = x16[2*i32 + 0] <<  0;
     x32     |= x16[2*i32 + 1] << 16;
@@ -24,45 +31,59 @@ static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32
     return x32;
 }
 
+// GFX906: get_int_b2_fast is defined in gfx906/gfx906-vecdotq.cuh as gfx906_get_int_b2_fast
+
 static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
-    return ((const int *) x)[i32]; // assume at least 4 byte alignment
+    return ((const int *) x)[i32];
+}
+
+static __device__ __forceinline__ int2 get_int_from_mxfp4_table(const uint32_t q4) {
+#if defined(GGML_USE_HIP) && defined(__gfx906__)
+    // GFX906: Use optimized lookup from gfx906-vecdotq.cuh
+    return gfx906_get_int_from_mxfp4_table(q4);
+#else
+    const int      q0_32  = (q4 >> 0) & 0x0F0F0F0F;
+    const int8_t * q0_8   = (const int8_t *) &q0_32;
+    const char4    val0_8 = make_char4(
+        kvalues_mxfp4[q0_8[0]], kvalues_mxfp4[q0_8[1]], kvalues_mxfp4[q0_8[2]], kvalues_mxfp4[q0_8[3]]);
+
+    const int      q1_32  = (q4 >> 4) & 0x0F0F0F0F;
+    const int8_t * q1_8   = (const int8_t *) &q1_32;
+    const char4    val1_8 = make_char4(
+        kvalues_mxfp4[q1_8[0]], kvalues_mxfp4[q1_8[1]], kvalues_mxfp4[q1_8[2]], kvalues_mxfp4[q1_8[3]]);
+
+    return make_int2(*((const int *) &val0_8), *((const int *) &val1_8));
+#endif
 }
 
-// q4 contains 8 indices with 4 bit each.
-// This function selects those bytes from table that are at those indices and returns them as int2.
-// The first int contains the bytes with even indices in q4, the second int contains the bytes with odd indices in q4.
 static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
 #if defined(GGML_USE_HIP)
-    // Load the 16-byte table into four 32-bit unsigned integers.
     const uint32_t *values = (const uint32_t *)table;
 
     const uint32_t q_even = q4;
     const uint32_t q_odd  = (q4 >> 4);
 
-    // Perform lookups in the lower half of the table (indices 0-7).
-    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
-    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);
+    const uint32_t sel_even = q_even & 0x07070707;
+    const uint32_t sel_odd  = q_odd & 0x07070707;
+
+    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], sel_even);
+    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], sel_odd);
+    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], sel_even);
+    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], sel_odd);
+
+    uint32_t b3e = (q_even >> 3) & 0x01010101;
+    uint32_t me = b3e; me |= me << 1; me |= me << 2; me |= me << 4;
 
-    // Perform lookups in the upper half of the table (indices 8-15).
-    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
-    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707);
+    uint32_t b3o = (q_odd >> 3) & 0x01010101;
+    uint32_t mo = b3o; mo |= mo << 1; mo |= mo << 2; mo |= mo << 4;
 
-    // Select between the low and high results based on the MSB of each index nibble.
-    uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
-    uint32_t res_x = __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even);
-    uint32_t mask_odd = 0x03020100 | ((q_odd & 0x08080808) >> 1);
-    uint32_t res_y = __builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd);
+    uint32_t res_x = (v_even_high & me) | (v_even_low & ~me);
+    uint32_t res_y = (v_odd_high & mo) | (v_odd_low & ~mo);
 
     return make_int2(res_x, res_y);
 #elif !defined(GGML_USE_MUSA)
-    // CUDA does not have an instruction for selecting bytes with 4 bit indices.
-    // However, __byte_perm is an instruction that selects bytes with 3 bit indices that can be used instead.
     const uint32_t * table32 = (const uint32_t *) table;
 
-    // __byte_perm selects bytes based on the lower 16 bits in its third argument.
-    // Therefore, do 2 iterations over the 32 bits in q4 with 0 and 16 shift.
-    // To handle the fourth bit, first call _byte_perm both for the low and the high 64 bit of table, using the low 3 bits.
-    // Then, call __byte_perm again to select from the low and high bytes based on the fourth bit.
     uint32_t tmp[2];
     const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
 #pragma unroll
@@ -74,12 +95,8 @@ static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, con
         tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
     }
 
-    // tmp contains the bytes from tyble in the same order as the 4 bit indices in q4.
-    // However, for the result we need ints with all even/odd 4 bit indices in q4.
-    // Therefore, 2 more calls to __byte_perm to put the bytes in the correct order.
     return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
 #else
-    // Generic implementation.
     const int      q0_32  = (q4 >> 0) & 0x0F0F0F0F;
     const int8_t * q0_8   = (const int8_t *) &q0_32;
     const char4    val0_8 = make_char4(
@@ -94,9 +111,6 @@ static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, con
 #endif
 }
 
-// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
-// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q
-
 #define VDR_Q4_0_Q8_1_MMVQ 2
 #define VDR_Q4_0_Q8_1_MMQ  4
 
@@ -110,14 +124,12 @@ template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_imp
         const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
         const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
 
-        // SIMD dot product of quantized values
         sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
         sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
     }
 
     const float2 ds8f = __half22float2(ds8);
 
-    // second part effectively subtracts 8 from each quant value
     return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
 }
 
@@ -134,7 +146,6 @@ template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_imp
         const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
         const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
 
-        // SIMD dot product of quantized values
         sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
         sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
     }
@@ -148,9 +159,7 @@ template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_imp
     const float2 ds8f = __half22float2(ds8);
     const float d4d8 = dm4f.x * ds8f.x;
     const float m4s8 = dm4f.y * ds8f.y;
-#endif // FAST_FP16_AVAILABLE
-
-    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
+#endif
     return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
 }
 
@@ -164,24 +173,23 @@ template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_imp
 
 #pragma unroll
     for (int i = 0; i < vdr; ++i) {
-        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
-        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
-        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
-        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
-        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
-        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values
-
-        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
-        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
-        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
-        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
-        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
-        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
+        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F;
+        vi0    |= (vh[i] <<  4) & 0x00000010;
+        vi0    |= (vh[i] << 11) & 0x00001000;
+        vi0    |= (vh[i] << 18) & 0x00100000;
+        vi0    |= (vh[i] << 25) & 0x10000000;
+        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
+
+        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F;
+        vi1    |= (vh[i] >> 12) & 0x00000010;
+        vi1    |= (vh[i] >>  5) & 0x00001000;
+        vi1    |= (vh[i] <<  2) & 0x00100000;
+        vi1    |= (vh[i] <<  9) & 0x10000000;
+        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
     }
 
     const float2 ds8f = __half22float2(ds8);
 
-    // second part effectively subtracts 16 from each quant value
     return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
 }
 
@@ -195,19 +203,19 @@ template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_imp
 
 #pragma unroll
     for (int i = 0; i < vdr; ++i) {
-        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
-        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
-        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
-        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
-        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
-        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values
-
-        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
-        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
-        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
-        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
-        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
-        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
+        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F;
+        vi0    |= (vh[i] <<  4) & 0x00000010;
+        vi0    |= (vh[i] << 11) & 0x00001000;
+        vi0    |= (vh[i] << 18) & 0x00100000;
+        vi0    |= (vh[i] << 25) & 0x10000000;
+        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
+
+        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F;
+        vi1    |= (vh[i] >> 12) & 0x00000010;
+        vi1    |= (vh[i] >>  5) & 0x00001000;
+        vi1    |= (vh[i] <<  2) & 0x00100000;
+        vi1    |= (vh[i] <<  9) & 0x10000000;
+        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
     }
 
 #ifdef FAST_FP16_AVAILABLE
@@ -219,9 +227,7 @@ template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_imp
     const float2 ds8f = __half22float2(ds8);
     const float d5d8 = dm5f.x * ds8f.x;
     const float m5s8 = dm5f.y * ds8f.y;
-#endif // FAST_FP16_AVAILABLE
-
-    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
+#endif
     return sumi*d5d8 + m5s8 / (QI5_1 / vdr);
 }
 
@@ -235,7 +241,6 @@ template <typename T, int vdr> static __device__ __forceinline__ T vec_dot_q8_0_
 
 #pragma unroll
     for (int i = 0; i < vdr; ++i) {
-        // SIMD dot product of quantized values
         sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
     }
 
@@ -249,7 +254,6 @@ template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_imp
 
 #pragma unroll
     for (int i = 0; i < vdr; ++i) {
-        // SIMD dot product of quantized values
         sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
     }
 
@@ -262,9 +266,7 @@ template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_imp
     const float2 ds8f = __half22float2(ds8);
     const float d8d8 = dm8f.x * ds8f.x;
     const float m8s8 = dm8f.y * ds8f.y;
-#endif // FAST_FP16_AVAILABLE
-
-    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
+#endif
     return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
 }
 
@@ -279,7 +281,6 @@ template <int vdr> static __device__ __forceinline__ float vec_dot_q8_0_16_q8_1_
 
 #pragma unroll
         for (int i = i0; i < i0 + QI8_0/2; ++i) {
-            // SIMD dot product of quantized values
             sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
         }
 
@@ -297,17 +298,22 @@ static __device__ __forceinline__ float vec_dot_mxfp4_q8_1(
 
     const block_mxfp4 * bq4 = (const block_mxfp4 *) vbq + kbx;
 
+#if defined(GGML_USE_HIP) && defined(__gfx906__)
+    // GFX906: Use software pipelined version from gfx906-vecdotq.cuh
+    int sumi = 0;
+    GFX906_VEC_DOT_MXFP4_Q8_1(bq4, bq8_1, iqs, sumi);
+#else
     const int * q8 = (const int *) bq8_1->qs + iqs;
-
     int sumi = 0;
 #pragma unroll
     for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
         const int aux_q4 = get_int_b1(bq4->qs, iqs + l);
-        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
+        const int2 v = get_int_from_mxfp4_table(aux_q4);
 
         sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
         sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
     }
+#endif
 
     const float d = ggml_cuda_e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds);
     return d * sumi;
@@ -316,7 +322,6 @@ static __device__ __forceinline__ float vec_dot_mxfp4_q8_1(
 #define VDR_Q2_K_Q8_1_MMVQ 1
 #define VDR_Q2_K_Q8_1_MMQ  4
 
-// contiguous v/x values
 static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
     const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
     const half2 & dm2, const float * __restrict__ d8) {
@@ -330,13 +335,12 @@ static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
 
         const int vi = (v >> (2*i)) & 0x03030303;
 
-        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product
+        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF));
 
-        // fill int with 4x m
         int m = sc >> 4;
         m |= m <<  8;
         m |= m << 16;
-        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
+        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0);
     }
 
     const float2 dm2f = __half22float2(dm2);
@@ -344,7 +348,6 @@ static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
     return dm2f.x*sumf_d - dm2f.y*sumf_m;
 }
 
-// contiguous v/x + u/y values
 template <int ns8>
 static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
     const int * __restrict__ v, const int * __restrict__ u, const half2 * dm2, const float & d8, const half2 * s8) {
@@ -399,7 +402,6 @@ static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
 #define VDR_Q3_K_Q8_1_MMVQ 1
 #define VDR_Q3_K_Q8_1_MMQ  2
 
-// contiguous v/x values
 static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
     const int & vl, const int & vh, const int * __restrict__ u, const uint8_t * __restrict__ scales,
     const int & scale_offset, const float & d3, const float * __restrict__ d8) {
@@ -426,13 +428,13 @@ static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
 
         const int vi = __vsubss4(vil, vih);
 
-        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
+        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
+
     }
 
     return d3 * sumf;
 }
 
-// contiguous v/x + u/y values
 static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
     const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ scales,
     const float & d3, const float & d8) {
@@ -445,7 +447,7 @@ static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
 
 #pragma unroll
         for (int i = i0; i < i0 + QI8_1/2; ++i) {
-            sumi_sc = ggml_cuda_dp4a(v[i], u[i], sumi_sc); // SIMD dot product
+            sumi_sc = ggml_cuda_dp4a(v[i], u[i], sumi_sc);
         }
 
         sumi += sumi_sc * scales[i0 / (QI8_1/2)];
@@ -457,7 +459,6 @@ static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
 #define VDR_Q4_K_Q8_1_MMVQ 2
 #define VDR_Q4_K_Q8_1_MMQ  8
 
-// contiguous v/x values
 static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
     const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
     const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {
@@ -470,11 +471,11 @@ static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
         const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
         const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;
 
-        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
-        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0)); // sum of u
+        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0));
+        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0));
 
         sumf_d += d8[i] * (dot1 * sc[i]);
-        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
+        sumf_m += d8[i] * (dot2 * m[i]);
     }
 
     const float2 dm4f = __half22float2(dm4);
@@ -482,7 +483,6 @@ static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
     return dm4f.x*sumf_d - dm4f.y*sumf_m;
 }
 
-// contiguous v/x + u/y values
 static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
     const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
     const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {
@@ -496,13 +496,13 @@ static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
 
 #pragma unroll
         for (int j = 0; j < QI8_1; ++j) {
-            sumi_d = ggml_cuda_dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d); // SIMD dot product
+            sumi_d = ggml_cuda_dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d);
         }
 
         const float2 ds8f = __half22float2(ds8[i]);
 
         sumf_d += ds8f.x * (sc[i] * sumi_d);
-        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
+        sumf_m += ds8f.y *   m[i];
     }
 
     const float2 dm4f = __half22float2(dm4);
@@ -513,7 +513,6 @@ static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
 #define VDR_Q5_K_Q8_1_MMVQ 2
 #define VDR_Q5_K_Q8_1_MMQ  8
 
-// contiguous v/x values
 static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
     const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
     const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {
@@ -532,8 +531,8 @@ static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
         const int v0i = vl0i | vh0i;
         const int v1i = vl1i | vh1i;
 
-        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
-        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0)); // sum of u
+        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0));
+        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0));
 
         sumf_d += d8[i] * (dot1 * sc[i]);
         sumf_m += d8[i] * (dot2 * m[i]);
@@ -545,7 +544,6 @@ static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
     return dm5f.x*sumf_d - dm5f.y*sumf_m;
 }
 
-// contiguous v/x + u/y values
 static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
     const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
     const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {
@@ -559,13 +557,13 @@ static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
 
 #pragma unroll
         for (int j = 0; j < QI8_1; ++j) {
-            sumi_d = ggml_cuda_dp4a(v[i*QI8_1 + j], u[i*QI8_1 + j], sumi_d); // SIMD dot product
+            sumi_d = ggml_cuda_dp4a(v[i*QI8_1 + j], u[i*QI8_1 + j], sumi_d);
         }
 
         const float2 ds8f = __half22float2(ds8[i]);
 
         sumf_d += ds8f.x * (sc[i] * sumi_d);
-        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
+        sumf_m += ds8f.y *   m[i];
     }
 
     const float2 dm4f = __half22float2(dm4);
@@ -576,7 +574,6 @@ static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
 #define VDR_Q6_K_Q8_1_MMVQ 1
 #define VDR_Q6_K_Q8_1_MMQ  8
 
-// contiguous v/x values
 static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
     const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
     const float & d, const float * __restrict__ d8) {
@@ -591,15 +588,14 @@ static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
 
         const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
 
-        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32
+        const int vi = __vsubss4((vil | vih), 0x20202020);
 
-        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
+        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
     }
 
     return d*sumf;
 }
 
-// contiguous v/x + u/y values
 static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
     const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
     const float & d6, const float * __restrict__ d8) {
@@ -611,15 +607,15 @@ static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
 
 #pragma unroll
     for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
-        int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale
+        int2 sumi_d = {0, 0};
 
 #pragma unroll
         for (int i = i0; i < i0 + 2; ++i) {
-            sumi_d.x = ggml_cuda_dp4a(v[2*i+0], u[2*i+0], sumi_d.x); // SIMD dot product
-            sumi_d.x = ggml_cuda_dp4a(v[2*i+1], u[2*i+1], sumi_d.x); // SIMD dot product
+            sumi_d.x = ggml_cuda_dp4a(v[2*i+0], u[2*i+0], sumi_d.x);
+            sumi_d.x = ggml_cuda_dp4a(v[2*i+1], u[2*i+1], sumi_d.x);
 
-            sumi_d.y = ggml_cuda_dp4a(v[2*i+4], u[2*i+4], sumi_d.y); // SIMD dot product
-            sumi_d.y = ggml_cuda_dp4a(v[2*i+5], u[2*i+5], sumi_d.y); // SIMD dot product
+            sumi_d.y = ggml_cuda_dp4a(v[2*i+4], u[2*i+4], sumi_d.y);
+            sumi_d.y = ggml_cuda_dp4a(v[2*i+5], u[2*i+5], sumi_d.y);
         }
 
         sumf_d += d8[i0/4] * (sc_reg[i0/2+0]*sumi_d.x + sc_reg[i0/2+1]*sumi_d.y);
@@ -715,7 +711,11 @@ static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
 
 #pragma unroll
     for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
+#if defined(GGML_USE_HIP) && defined(__gfx906__)
+        v[i] = gfx906_get_int_b2_fast(bq8_0->qs, iqs + i);
+#else
         v[i] = get_int_b2(bq8_0->qs, iqs + i);
+#endif
         u[i] = get_int_b4(bq8_1->qs, iqs + i);
     }
 
@@ -757,7 +757,6 @@ static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
 
     const int vl = get_int_b2(bq3_K->qs, iqs);
 
-    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
     const int vh = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;
 
     int    u[QR3_K];
@@ -781,14 +780,8 @@ static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
     int    u[2*QR4_K];
     float d8[QR4_K];
 
-    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
     const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
 
-    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
-    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
-    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
-    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108
-
     const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
     v[0] = q4[0];
     v[1] = q4[4];
@@ -1050,7 +1043,6 @@ static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
 #define VDR_IQ3_S_Q8_1_MMVQ 2
 #define VDR_IQ3_S_Q8_1_MMQ  2
 
-// TODO: don't use lookup table for signs
 static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
     const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
 
diff --git a/ggml/src/ggml-cuda/vendors/hip.h b/ggml/src/ggml-cuda/vendors/hip.h
index 016b04e5a..5cc1b5431 100644
--- a/ggml/src/ggml-cuda/vendors/hip.h
+++ b/ggml/src/ggml-cuda/vendors/hip.h
@@ -138,6 +138,8 @@
 #define cudaStream_t hipStream_t
 #define cudaSuccess hipSuccess
 #define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
+#define cudaFuncSetAttribute hipFuncSetAttribute
+#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
 #define __trap() do { abort(); __builtin_unreachable(); } while(0)
 #define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
 #define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
