/**
 *  libBICOS: binary correspondence search on multishot stereo imagery
 *  Copyright (C) 2024-2025  Robotics Group @ Julius-Maximilian University
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "common.hpp"
#include "compat.hpp"
#include "impl/common.hpp"
#include "impl/cuda/cutil.cuh"

#include <opencv2/core/cuda/common.hpp>

namespace BICOS::impl::cuda {

enum class NXCVariant { PLAIN, MINVAR };

template<typename T, typename V>
using corrfun = V (*)(const T*, const T*, size_t, V);

template<NXCVariant VARIANT, typename T>
__device__ double nxcorrd(
    const T* __restrict__ pix0,
    const T* __restrict__ pix1,
    size_t n,
    [[maybe_unused]] double minvar
) {
    double mean0 = 0.0, mean1 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mean0 = __dadd_rn(mean0, pix0[i]);
        mean1 = __dadd_rn(mean1, pix1[i]);
    }

    mean0 = __ddiv_rn(mean0, n);
    mean1 = __ddiv_rn(mean1, n);

    double covar = 0.0, var0 = 0.0, var1 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff0 = pix0[i] - mean0, diff1 = pix1[i] - mean1;

        covar = __fma_rn(diff0, diff1, covar);
        var0 = __fma_rn(diff0, diff0, var0);
        var1 = __fma_rn(diff1, diff1, var1);
    }

    if constexpr (NXCVariant::MINVAR == VARIANT)
        if (var0 < minvar || var1 < minvar)
            return -1.0;

    return covar / sqrt(var0 * var1);
}

template<NXCVariant VARIANT, typename T>
__device__ float nxcorrf(
    const T* __restrict__ pix0,
    const T* __restrict__ pix1,
    size_t n,
    [[maybe_unused]] float minvar
) {
    float mean0 = 0.0f, mean1 = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        mean0 = __fadd_rn(mean0, pix0[i]);
        mean1 = __fadd_rn(mean1, pix1[i]);
    }

    mean0 = __fdiv_rn(mean0, n);
    mean1 = __fdiv_rn(mean1, n);

    float covar = 0.0f, var0 = 0.0f, var1 = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff0 = pix0[i] - mean0, diff1 = pix1[i] - mean1;

        covar = __fmaf_rn(diff0, diff1, covar);
        var0 = __fmaf_rn(diff0, diff0, var0);
        var1 = __fmaf_rn(diff1, diff1, var1);
    }

    if constexpr (NXCVariant::MINVAR == VARIANT)
        if (var0 < minvar || var1 < minvar)
            return -1.0f;

    return covar / sqrtf(var0 * var1);
}

using agree_kernel_t = void (*)(
    const cv::cuda::PtrStepSz<int16_t> raw_disp,
    const GpuMatHeader* _stacks,
    size_t n,
    double min_nxc,
    double min_var,
    float subpixel_step,
    cv::cuda::PtrStepSz<float> out,
    [[maybe_unused]] GpuMatHeader _corrmap
);

template<typename TInput, typename TPrecision, NXCVariant VARIANT, bool CORRMAP, size_t NPIX>
__global__ void agree_kernel(
    cv::cuda::PtrStepSz<int16_t> raw_disp,
    const GpuMatHeader* stacks,
    size_t n,
    double min_nxc,
    double min_var,
    [[maybe_unused]] float,
    [[maybe_unused]] cv::cuda::PtrStepSz<float>,
    [[maybe_unused]] GpuMatHeader corrmap
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (raw_disp.cols <= col || raw_disp.rows <= row)
        return;

    int16_t& d = raw_disp(row, col);

    if (is_invalid(d))
        return;

    const int col1 = col - d;

    if UNLIKELY (col1 < 0 || raw_disp.cols <= col1) {
        d = INVALID_DISP<int16_t>;
        return;
    }

    TInput pix0[NPIX], pix1[NPIX];

    const GpuMatHeader *stack0 = stacks, *stack1 = stacks + n;

    for (size_t t = 0; t < n; ++t)
        pix0[t] = load_datacache(stack0[t].ptr<TInput>(row) + col);
    for (size_t t = 0; t < n; ++t)
        pix1[t] = load_datacache(stack1[t].ptr<TInput>(row) + col1);

    TPrecision nxc;
    if constexpr (std::is_same_v<TPrecision, float>)
        nxc = nxcorrf<VARIANT>(pix0, pix1, n, min_var);
    else
        nxc = nxcorrd<VARIANT>(pix0, pix1, n, min_var);

    if constexpr (CORRMAP)
        corrmap.at<TPrecision>(row, col) = nxc;

    if (nxc < (TPrecision)min_nxc)
        d = INVALID_DISP<int16_t>;
}

template<typename TInput, typename TPrecision, NXCVariant VARIANT, bool CORRMAP, size_t NPIX>
__global__ void agree_subpixel_kernel(
    cv::cuda::PtrStepSz<int16_t> _raw_disp,
    const GpuMatHeader* stacks,
    size_t n,
    double min_nxc,
    double min_var,
    float subpixel_step,
    cv::cuda::PtrStepSz<float> out,
    [[maybe_unused]] GpuMatHeader corrmap
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out.cols <= col || out.rows <= row)
        return;

    const cv::cuda::PtrStepSz<int16_t> raw_disp = _raw_disp;

    const int16_t d = load_datacache(raw_disp.ptr(row) + col);

    if (is_invalid(d))
        return;

    const int col1 = col - d;

    if UNLIKELY (col1 < 0 || out.cols <= col1)
        return;

    TInput pix0[NPIX], pix1[NPIX];

    const GpuMatHeader *stack0 = stacks, *stack1 = stacks + n;

    for (size_t t = 0; t < n; ++t)
        pix0[t] = load_datacache(stack0[t].ptr<TInput>(row) + col);
    for (size_t t = 0; t < n; ++t)
        pix1[t] = load_datacache(stack1[t].ptr<TInput>(row) + col1);

    TPrecision nxc;

    if UNLIKELY (col1 == 0 || col1 == out.cols - 1) {
        if constexpr (std::is_same_v<TPrecision, float>)
            nxc = nxcorrf<VARIANT>(pix0, pix1, n, min_var);
        else
            nxc = nxcorrd<VARIANT>(pix0, pix1, n, min_var);

        if constexpr (CORRMAP)
            corrmap.at<TPrecision>(row, col) = nxc;

        if (nxc < (TPrecision)min_nxc)
            return;

        out(row, col) = d;
    } else {
        TInput interp[NPIX];
        float a[NPIX], b[NPIX], c[NPIX];

        // clang-format off

        for (size_t t = 0; t < n; ++t) {
            TInput y0 = load_datacache(stack1[t].ptr<TInput>(row) + col1 - 1),
                   y1 = pix1[t],
                   y2 = load_datacache(stack1[t].ptr<TInput>(row) + col1 + 1);

            a[t] = 0.5f * ( y0 - 2.0f * y1 + y2);
            b[t] = 0.5f * (-y0             + y2);
            c[t] = y1;
        }

        float best_x = 0.0f;
        TPrecision best_nxc = -1.0;

        for (float x = -1.0f; x <= 1.0f; x += subpixel_step) {
            for (size_t t = 0; t < n; ++t)
                interp[t] = (TInput)__float2int_rn(a[t] * x * x + b[t] * x + c[t]);

            if constexpr (std::is_same_v<TPrecision, float>)
                nxc = nxcorrf<VARIANT>(pix0, interp, n, min_var);
            else
                nxc = nxcorrd<VARIANT>(pix0, interp, n, min_var);

            if (best_nxc < nxc) {
                best_x = x;
                best_nxc = nxc;
            }
        }

        if constexpr (CORRMAP)
            corrmap.at<TPrecision>(row, col) = best_nxc;

        if (best_nxc < (TPrecision)min_nxc)
            return;

        // larger x -> further to the right -> less disparity
        out(row, col) = d - best_x;

        // clang-format on 
    }
}

} // namespace BICOS::impl::cuda
