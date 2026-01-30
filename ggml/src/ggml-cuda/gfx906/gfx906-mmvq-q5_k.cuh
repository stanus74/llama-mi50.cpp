#pragma once

// GFX906 Warp-Cooperative Q5_K GEMV Kernel
// Uses half-warp (32 threads) per row for better memory coalescing
// Designed for small matrices (ncols <= 1024)

#if defined(GGML_USE_HIP)

__launch_bounds__(64, 1)
static __global__ void gfx906_mul_mat_vec_q5_K_warp_coop(
        const void * __restrict__ vx, const void * __restrict__ vy,
        const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y,
        const uint32_t stride_row_x,
        const uint32_t stride_col_dst, const uint3 channel_ratio,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst, const uint32_t nrows_x) {

    constexpr int qk_q5_k = QK_K;
    constexpr int qi_q5_k = QI5_K;
    constexpr int vdr_q5_k = VDR_Q5_K_Q8_1_MMVQ;

    const int lane_id = threadIdx.x;
    const int half_lane = lane_id % 32;
    const int row_offset = lane_id / 32;

    const int row = blockIdx.x * 2 + row_offset;
    if (row >= (int)nrows_x) return;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t channel_x   = ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    const int blocks_per_row = ncols_x / qk_q5_k;
    const int kbx_offset = sample_x * stride_sample_x + channel_x * stride_channel_x + row * stride_row_x;

    const block_q5_K * x = (const block_q5_K *)vx + kbx_offset;
    const block_q8_1 * y = (const block_q8_1 *)vy + sample_y * stride_sample_y + channel_y * stride_channel_y;

    float sumf = 0.0f;

    constexpr int q8_blocks_per_q5 = qk_q5_k / QK8_1;
    constexpr int lanes_per_block = qi_q5_k / vdr_q5_k;

    for (int ib = 0; ib < blocks_per_row; ++ib) {
        float partial = 0.0f;
        if (half_lane < lanes_per_block) {
            const int iqs = vdr_q5_k * half_lane;
            const int kby = ib * q8_blocks_per_q5;
            partial = vec_dot_q5_K_q8_1(x + ib, y + kby, iqs);
        }

        partial = warp_reduce_sum<32>(partial);

        if (half_lane == 0) {
            sumf += partial;
        }
    }

    if (half_lane == 0) {
        dst[sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row] = sumf;
    }
}

static void gfx906_launch_mul_mat_vec_q5_K_warp_coop(
        const void * vx, const void * vy, const int32_t * ids,
        float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y,
        const uint32_t stride_row_x,
        const uint32_t stride_col_dst, const uint3 channel_ratio,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst, const uint32_t nrows_x,
        const uint32_t nchannels_dst, const uint32_t nsamples_dst,
        cudaStream_t stream) {

    const dim3 block_dims(64, 1, 1);
    const dim3 block_nums((nrows_x + 1) / 2, nchannels_dst, nsamples_dst);

    gfx906_mul_mat_vec_q5_K_warp_coop<<<block_nums, block_dims, 0, stream>>>(
        vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x,
        stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y,
        stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
        stride_sample_dst, nrows_x);
}

#endif // GGML_USE_HIP
