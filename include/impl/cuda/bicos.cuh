#pragma once

#include "config.hpp"
#include "stepbuf.hpp"

namespace BICOS::impl::cuda {

static __device__ __forceinline__ int ham(uint32_t a, uint32_t b) {
    return __builtin_popcount(a ^ b);
}

static __device__ __forceinline__ int ham(uint64_t a, uint64_t b) {
    return __builtin_popcountll(a ^ b);
}

static __device__ __forceinline__ int ham(uint128_t a, uint128_t b) {
    const uint128_t diff = a ^ b;
    int lo = __builtin_popcountll((uint64_t)(diff & 0xFFFFFFFFFFFFFFFFUL));
    int hi = __builtin_popcountll((uint64_t)(diff >> 64));
    return lo + hi;
}

template<typename TDescriptor>
__global__ void bicos_kernel(
    const StepBuf<TDescriptor>* descr0,
    const StepBuf<TDescriptor>* descr1,
    cv::cuda::PtrStepSz<int16_t> out
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out.rows <= row)
        return;

    extern __shared__ char _row1[];
    TDescriptor* row1 = (TDescriptor*)_row1;

    for (size_t i = threadIdx.x; i < out.cols; i += blockDim.x)
        row1[i] = descr1->row(row)[i];

    if (out.cols <= col)
        return;

    __syncthreads();

    const TDescriptor d0 = descr0->row(row)[col];

    int best_col1 = -1, min_cost = INT_MAX, num_duplicate_minima = 0;

    for (size_t col1 = 0; col1 < out.cols; ++col1) {
        const TDescriptor d1 = row1[col1];

        int cost = ham(d0, d1);

        if (cost < min_cost) {
            min_cost = cost;
            best_col1 = col1;
            num_duplicate_minima = 0;
        } else if (cost == min_cost) {
            num_duplicate_minima++;
        }
    }

    if (0 < num_duplicate_minima)
        return;

    out(row, col) = abs(col - best_col1);
}

} // namespace BICOS::impl
