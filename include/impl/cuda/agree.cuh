#pragma once

#include "common.hpp"

#include <opencv2/core/cuda/common.hpp>

namespace BICOS::impl::cuda {

template<typename T>
static __device__ double nxcorr(const T* pix0, const T* pix1, size_t n) {
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

    return n_expectancy / sqrt(sqdiffsum0 * sqdiffsum1);
}

template<typename TInput>
__global__ void agree_kernel(
    const cv::cuda::PtrStepSz<int16_t> raw_disp,
    const cv::cuda::PtrStepSz<TInput>* stacks,
    size_t n,
    double nxcorr_threshold,
    cv::cuda::PtrStepSz<disparity_t> out
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out.cols <= col || out.rows <= row)
        return;

    const int16_t d = raw_disp(row, col);

    if (d == INVALID_DISP_<int16_t>)
        return;

    int col1 = col - d;

    if (col1 < 0 || out.cols <= col1)
        return;

    TInput pix0[33], pix1[33];

    const cv::cuda::PtrStepSz<TInput>*stack0 = stacks, *stack1 = stacks + n;

    for (size_t t = 0; t < n; ++t) {
        pix0[t] = stack0[t](row, col);
        pix1[t] = stack1[t](row, col1);
#ifdef BICOS_DEBUG
        if (t >= 33)
            __trap();
#endif
    }

    double nxc = nxcorr(pix0, pix1, n);

    if (nxc < nxcorr_threshold)
        return;

    out(row, col) = d;
}

template<typename TInput>
__global__ void agree_subpixel_kernel(
    const cv::cuda::PtrStepSz<int16_t> raw_disp,
    const cv::cuda::PtrStepSz<TInput>* stacks,
    size_t n,
    double nxcorr_threshold,
    float subpixel_step,
    cv::cuda::PtrStepSz<disparity_t> out
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out.cols <= col || out.rows <= row)
        return;

    const int16_t d = raw_disp(row, col);

    if (d == INVALID_DISP_<int16_t>)
        return;

    const int col1 = col - d;

    if (col1 < 0 || out.cols <= col1)
        return;

    TInput pix0[33], pix1[33];

    const cv::cuda::PtrStepSz<TInput>
        *stack0 = stacks,
        *stack1 = stacks + n;

    for (size_t t = 0; t < n; ++t) {
        pix0[t] = stack0[t](row, col);
        pix1[t] = stack1[t](row, col1);
#ifdef BICOS_DEBUG
        if (t >= 33)
            __trap();
#endif
    }

    if (col1 == 0 || col1 == out.cols - 1) {
        double nxc = nxcorr(pix0, pix1, n);

        if (nxc < nxcorr_threshold)
            return;

        out(row, col) = d;
    } else {
        TInput interp[33];
        float a[33], b[33], c[33];

        // clang-format off

        for (size_t t = 0; t < n; ++t) {
            TInput y0 = stack1[t](row, col1 - 1), 
                   y1 = pix1[t],
                   y2 = stack1[t](row, col1 + 1);

            a[t] = 0.5f * ( y0 - 2.0f * y1 + y2);
            b[t] = 0.5f * (-y0             + y2);
            c[t] = y1;
        }

        float best_x = 0.0f;
        double best_nxcorr = -1.0;

        for (float x = -1.0f; x <= 1.0f; x += subpixel_step) {
            for (size_t t = 0; t < n; ++t)
                interp[t] = (TInput)__float2int_rn(a[t] * x * x + b[t] * x + c[t]);

            double nxc = nxcorr(pix0, interp, n);

            if (best_nxcorr < nxc) {
                best_x = x;
                best_nxcorr = nxc;
            }
        }

        if (best_nxcorr < nxcorr_threshold)
            return;

        out(row, col) = d + best_x;

        // clang-format on 
    }
}

template<typename TInput>
__global__ void agree_subpixel_kernel_smem(
    const cv::cuda::PtrStepSz<int16_t> raw_disp,
    const cv::cuda::PtrStepSz<TInput>* stacks,
    size_t n,
    double nxcorr_threshold,
    float subpixel_step,
    cv::cuda::PtrStepSz<disparity_t> out
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out.rows <= row)
        return;

    extern __shared__ char _rows[];
    TInput *row1 = (TInput*)_rows;
    const cv::cuda::PtrStepSz<TInput>
        *stack0 = stacks,
        *stack1 = stacks + n;

    for (size_t c = threadIdx.x; c < out.cols; c += blockDim.x)
        for (size_t t = 0; t < n; ++t)
            row1[c * n + t] = stack1[t](row, c);

    if (out.cols <= col)
        return;

    __syncthreads();

    const int16_t d = raw_disp(row, col);

    if (d == INVALID_DISP_<int16_t>)
        return;

    const int col1 = col - d;

    if (col1 < 0 || out.cols <= col1)
        return;

    TInput pix0[33];
    for (size_t t = 0; t < n; ++t) {
        pix0[t] = stack0[t](row, col);
#ifdef BICOS_DEBUG
        if (t >= 33)
            __trap();
#endif
    }

    if (col1 == 0 || col1 == out.cols - 1) {
        double nxc = nxcorr(pix0, row1 + n * col1, n);

        if (nxc < nxcorr_threshold)
            return;

        out(row, col) = d;
    } else {
        TInput interp[33];
        float a[33], b[33], c[33];

        // clang-format off

        for (size_t t = 0; t < n; ++t) {
            TInput y0 = row1[n * col1 - 1 + t], 
                   y1 = row1[n * col1     + t],
                   y2 = row1[n * col1 + 1 + t];

            a[t] = 0.5f * ( y0 - 2.0f * y1 + y2);
            b[t] = 0.5f * (-y0             + y2);
            c[t] = y1;
        }

        float best_x = 0.0f;
        double best_nxcorr = -1.0;

        for (float x = -1.0f; x <= 1.0f; x += subpixel_step) {
            for (size_t t = 0; t < n; ++t)
                interp[t] = (TInput)__float2int_rn(a[t] * x * x + b[t] * x + c[t]);

            double nxc = nxcorr(pix0, interp, n);

            if (best_nxcorr < nxc) {
                best_x = x;
                best_nxcorr = nxc;
            }
        }

        if (best_nxcorr < nxcorr_threshold)
            return;

        out(row, col) = d + best_x;

        // clang-format on 
    }
}

} // namespace BICOS::impl::cuda
