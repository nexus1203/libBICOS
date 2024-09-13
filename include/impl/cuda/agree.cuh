#pragma once

#include "../util.hpp"
#include "config.hpp"

namespace BICOS::impl::cuda {

template<typename T>
static __forceinline__ __device__ double nxcorr(const T* pix0, const T* pix1, size_t n) {
    double mean0 = 0.0, mean1 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mean0 += pix0[i];
        mean1 += pix1[i];
    }
    mean0 /= double(n);
    mean1 /= double(n);

    double n_expectancy = 0.0, sqdiffsum0 = 0.0, sqdiffsum1 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff0 = pix0[i] - mean0, diff1 = pix1[i] - mean1;

        n_expectancy += diff0 * diff1;
        sqdiffsum0 += diff0 * diff0;
        sqdiffsum1 += diff1 * diff1;
    }

    return n_expectancy / std::sqrt(sqdiffsum0 * sqdiffsum1);
}

template<typename TInput>
__global__ void agree_kernel(
    const cv::cuda::PtrStepSz<int16_t> raw_disp,
    const cv::cuda::PtrStepSz<TInput>* stacks,
    size_t n,
    cv::Size sz,
    double nxcorr_threshold,
    cv::cuda::PtrStepSz<disparity_t> out
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out.cols <= x || out.rows <= y) {
        // __syncthreads();
        return;
    }

    int16_t d0 = raw_disp(y, x);

    if (d0 == INVALID_DISP_<int16_t>) {
        // __syncthreads();
        return;
    }

    TInput *pix0 = STACKALLOC(n, TInput), *pix1 = STACKALLOC(n, TInput);

    for (size_t t = 0; t < n; ++t) {
        pix0[t] = stacks[t](y, x);
        pix1[t] = stacks[n + t](y, x);
    }

    double nxc = nxcorr(pix0, pix1, n);

    if (nxc < nxcorr_threshold)
        return;

    out(y, x) = d0;

    /* prefetch memory? */
    /*

    extern __shared__ char _rows[];
    TInput *row0 = (TInput*)_rows;
    

    for (size_t i = threadIdx.x * blockDim.x; i < out.cols && i < (threadIdx.x + 1) * blockDim.x; ++i) {
        for (size_t t = 0; t < n; ++t) {
            row0[i * n + t] = stacks[n + t](y, i);
        }
    }

    for (size_t t = 0; t < n; ++t)
        pix0[t] = stacks[t](y, x);

    __syncthreads();

    */
}

template<typename TInput>
__global__ void agree_subpixel_kernel(
    const cv::cuda::PtrStepSz<int16_t> raw_disp,
    const cv::cuda::PtrStepSz<TInput>* stacks,
    size_t n,
    cv::Size sz,
    double nxcorr_threshold,
    float subpixel_step,
    cv::cuda::PtrStepSz<disparity_t> out
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out.cols <= x || out.rows <= y)
        return;

    const int16_t d0 = raw_disp(y, x);

    if (d0 == INVALID_DISP_<int16_t>)
        return;

    const int idx1 = x - d0;

    if (idx1 < 0 || sz.width <= idx1)
        return;

    TInput *pix0 = STACKALLOC(n, TInput), *pix1 = STACKALLOC(n, TInput);

    const cv::cuda::PtrStepSz<TInput>*stack0 = stacks, *stack1 = stacks + n;

    for (size_t t = 0; t < n; ++t) {
        pix0[t] = stack0[t](y, x);
        pix1[t] = stack1[t](y, x);
    }

    if (idx1 == 0 || idx1 == sz.width - 1) {
        double nxc = nxcorr(pix0, pix1, n);

        if (nxc < nxcorr_threshold)
            return;

        out(y, x) = d0;
    } else {
        TInput* interp = STACKALLOC(n, TInput);
        float *a = STACKALLOC(n, float), *b = STACKALLOC(n, float), *c = STACKALLOC(n, float);

        for (size_t t = 0; t < n; ++t) {
            TInput y0 = stack1[t](y, idx1 - 1), y1 = pix1[t], y2 = stack1[t](y, idx1 + 1);

            a[t] = 0.5f * (y0 - 2.0f * y1 + y2);
            b[t] = 0.5f * (-y0 + y2);
            c[t] = y1;
        }

        float best_x = 0.0f;
        double best_nxcorr = -1.0;

        for (float x1 = -1.0f; x1 <= 1.0f; x1 += subpixel_step) {
            for (size_t t = 0; t < n; ++t)
                interp[t] = TInput(a[t] * x * x + b[t] * x + c[t]);

            double nxc = nxcorr(pix0, interp, n);

            if (best_nxcorr < nxc) {
                best_x = x;
                best_nxcorr = nxc;
            }
        }

        if (best_nxcorr < nxcorr_threshold)
            return;

        out(y, x) = d0 + best_x;
    }
}

} // namespace BICOS::impl
