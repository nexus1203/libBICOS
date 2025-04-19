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

#include "common.hpp"
#include "cuda.hpp"

#include "impl/common.hpp"
#include "impl/cuda/agree.cuh"
#include "impl/cuda/bicos.cuh"
#include "impl/cuda/cutil.cuh"
#include "impl/cuda/descriptor_transform.cuh"

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <string>
#include <stdexcept>

namespace BICOS::impl::cuda {

template<typename TDescriptor>
static auto select_bicos_kernel(SearchVariant variant, bool smem) {
    if (std::holds_alternative<Variant::Consistency>(variant)) {
        auto consistency = std::get<Variant::Consistency>(variant);
        if (consistency.no_dupes)
            return smem                                          ? bicos_kernel_smem < TDescriptor,
                   BICOSFLAGS_CONSISTENCY | BICOSFLAGS_NODUPES > : bicos_kernel < TDescriptor,
                   BICOSFLAGS_CONSISTENCY | BICOSFLAGS_NODUPES > ;
        else
            return smem ? bicos_kernel_smem<TDescriptor, BICOSFLAGS_CONSISTENCY>
                        : bicos_kernel<TDescriptor, BICOSFLAGS_CONSISTENCY>;
    } else if (std::holds_alternative<Variant::NoDuplicates>(variant))
        return smem ? bicos_kernel_smem<TDescriptor, BICOSFLAGS_NODUPES>
                    : bicos_kernel<TDescriptor, BICOSFLAGS_NODUPES>;

    throw std::invalid_argument("unimplemented");
}

template<typename TInput, typename TDescriptor>
static void match_impl(
    const std::vector<cv::cuda::GpuMat>& _stack0,
    const std::vector<cv::cuda::GpuMat>& _stack1,
    size_t n_images,
    cv::Size sz,
    std::optional<float> min_nxc,
    Precision precision,
    TransformMode mode,
    std::optional<float> subpixel_step,
    std::optional<float> min_var,
    SearchVariant variant,
    cv::cuda::GpuMat& out,
    cv::cuda::GpuMat* corrmap,
    cv::cuda::Stream& _stream
) {
    std::vector<GpuMatHeader> ptrs_host(2 * n_images);

    for (size_t i = 0; i < n_images; ++i) {
        ptrs_host[i] = _stack0[i];
        ptrs_host[i + n_images] = _stack1[i];
    }

    StepBuf<TDescriptor> descr0(sz), descr1(sz);

    cudaStream_t mainstream = cv::cuda::StreamAccessor::getStream(_stream);

    /* descriptor transform */

    cudaStream_t substream0, substream1;
    assertCudaSuccess(cudaStreamCreate(&substream0));
    assertCudaSuccess(cudaStreamCreate(&substream1));

    cudaEvent_t event0, event1;
    assertCudaSuccess(cudaEventCreate(&event0));
    assertCudaSuccess(cudaEventCreate(&event1));

    RegisteredPtr ptrs_dev(ptrs_host.data(), 2 * n_images, true);
    RegisteredPtr descr0_dev(&descr0), descr1_dev(&descr1);

    int pix_idx;
    switch (n_images) {
    case 0 ... 4: pix_idx = 0; break;
    case 5 ... 8: pix_idx = 1; break;
    case 9 ... 12: pix_idx = 2; break;
    case 13 ... 16: pix_idx = 3; break;
    case 17 ... 20: pix_idx = 4; break;
    case 21 ... 25: pix_idx = 5; break;
    case 26 ... 30: pix_idx = 6; break;
    case 31 ... 45: pix_idx = 7; break;
    case 46 ... 65: pix_idx = 8; break;
    default: throw Exception("Bad number of images");
    }

    dim3 block, grid;

    /*[[[cog
    import cog
    cog.outl('constexpr static transform_kernel_t<TDescriptor> transform_lut[2][9] = {')
    for kernel in ['transform_full_kernel', 'transform_limited_kernel']:
        cog.outl('\t{')
        for npix in [4, 8, 12, 16, 20, 30, 35, 40, 65]:
            cog.outl(f'\t\t{kernel}<TInput, TDescriptor, {npix}>,')
        cog.outl('\t},')
    cog.outl('};')
    ]]]*/
    constexpr static transform_kernel_t<TDescriptor> transform_lut[2][9] = {
    	{
    		transform_full_kernel<TInput, TDescriptor, 4>,
    		transform_full_kernel<TInput, TDescriptor, 8>,
    		transform_full_kernel<TInput, TDescriptor, 12>,
    		transform_full_kernel<TInput, TDescriptor, 16>,
    		transform_full_kernel<TInput, TDescriptor, 20>,
    		transform_full_kernel<TInput, TDescriptor, 30>,
    		transform_full_kernel<TInput, TDescriptor, 35>,
    		transform_full_kernel<TInput, TDescriptor, 40>,
    		transform_full_kernel<TInput, TDescriptor, 65>,
    	},
    	{
    		transform_limited_kernel<TInput, TDescriptor, 4>,
    		transform_limited_kernel<TInput, TDescriptor, 8>,
    		transform_limited_kernel<TInput, TDescriptor, 12>,
    		transform_limited_kernel<TInput, TDescriptor, 16>,
    		transform_limited_kernel<TInput, TDescriptor, 20>,
    		transform_limited_kernel<TInput, TDescriptor, 30>,
    		transform_limited_kernel<TInput, TDescriptor, 35>,
    		transform_limited_kernel<TInput, TDescriptor, 40>,
    		transform_limited_kernel<TInput, TDescriptor, 65>,
    	},
    };
    //[[[end]]]

    transform_kernel_t<TDescriptor> transform_kernel = transform_lut[mode == TransformMode::LIMITED][pix_idx];
    block = max_blocksize(transform_kernel);
    grid = create_grid(block, sz);

    transform_kernel<<<grid, block, 0, substream0>>>(ptrs_dev, n_images, sz, descr0_dev);
    assertCudaSuccess(cudaGetLastError());
    assertCudaSuccess(cudaEventRecord(event0, substream0));

    transform_kernel<<<grid, block, 0, substream1>>>(ptrs_dev + n_images, n_images, sz, descr1_dev);
    assertCudaSuccess(cudaGetLastError());
    assertCudaSuccess(cudaEventRecord(event1, substream1));

    assertCudaSuccess(cudaStreamWaitEvent(mainstream, event0));
    assertCudaSuccess(cudaStreamWaitEvent(mainstream, event1));

    /* bicos disparity */

    cv::cuda::GpuMat bicos_disp;
    if (out.type() == cv::DataType<int16_t>::type) {
        // input buffer is probably output from previous,
        // non-subpixel call to match()
        // we can reuse that
        bicos_disp = out;
    }

    init_disparity<int16_t>(bicos_disp, sz, _stream);

    auto kernel = select_bicos_kernel<TDescriptor>(variant, true);
    int lr_max_diff = std::holds_alternative<Variant::Consistency>(variant)
        ? std::get<Variant::Consistency>(variant).max_lr_diff
        : -1;

    size_t smem_size = sz.width * sizeof(TDescriptor);
    bool bicos_smem_fits = cudaSuccess
        == cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaGetLastError(); // clear potential error from previous call to cudaFuncSetAttribute

    if (bicos_smem_fits) {
        block = max_blocksize(kernel, smem_size);
        grid = create_grid(block, sz);

        kernel<<<grid, block, smem_size, mainstream>>>(
            descr0_dev,
            descr1_dev,
            lr_max_diff,
            bicos_disp
        );
    } else {
        kernel = select_bicos_kernel<TDescriptor>(variant, false);

        block = max_blocksize(kernel);
        grid = create_grid(block, sz);

        kernel<<<grid, block, 0, mainstream>>>(descr0_dev, descr1_dev, lr_max_diff, bicos_disp);
    }
    assertCudaSuccess(cudaGetLastError());

    // optimized for subpixel interpolation.
    // keep `out` as output for the interpolated
    // depth map.

    if (!min_nxc.has_value()) {
        out = bicos_disp;
        return;
    }

    /* nxcorr */

    if (corrmap)
        corrmap->create(sz, precision == Precision::SINGLE ? CV_32FC1 : CV_64FC1);

    // clang-format off

    /*[[[cog
    import cog
    cog.outl('constexpr static agree_kernel_t agree_lut[2][2][2][2][9] = {')
    for kernel in ['agree_kernel', 'agree_subpixel_kernel']:
        cog.outl('\t{')
        for corrmap in ['false', 'true']:
            cog.outl('\t\t{')
            for variant in ['NXCVariant::PLAIN', 'NXCVariant::MINVAR']:
                cog.outl('\t\t\t{')
                for precision in ['float', 'double']:
                    cog.outl('\t\t\t\t{')
                    for npix in [4, 8, 12, 16, 20, 30, 35, 40, 65]:
                        cog.outl(f'\t\t\t\t{kernel}<TInput, {precision}, {variant}, {corrmap}, {npix}>,')
                    cog.outl('\t\t\t\t},')
                cog.outl('\t\t\t},')
            cog.outl('\t\t},')
        cog.outl('\t\t},')
    cog.outl('};')
    ]]]*/
    constexpr static agree_kernel_t agree_lut[2][2][2][2][9] = {
    	{
    		{
    			{
    				{
    				agree_kernel<TInput, float, NXCVariant::PLAIN, false, 4>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, false, 8>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, false, 12>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, false, 16>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, false, 20>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, false, 30>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, false, 35>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, false, 40>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, false, 65>,
    				},
    				{
    				agree_kernel<TInput, double, NXCVariant::PLAIN, false, 4>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, false, 8>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, false, 12>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, false, 16>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, false, 20>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, false, 30>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, false, 35>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, false, 40>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, false, 65>,
    				},
    			},
    			{
    				{
    				agree_kernel<TInput, float, NXCVariant::MINVAR, false, 4>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, false, 8>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, false, 12>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, false, 16>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, false, 20>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, false, 30>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, false, 35>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, false, 40>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, false, 65>,
    				},
    				{
    				agree_kernel<TInput, double, NXCVariant::MINVAR, false, 4>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, false, 8>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, false, 12>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, false, 16>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, false, 20>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, false, 30>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, false, 35>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, false, 40>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, false, 65>,
    				},
    			},
    		},
    		{
    			{
    				{
    				agree_kernel<TInput, float, NXCVariant::PLAIN, true, 4>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, true, 8>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, true, 12>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, true, 16>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, true, 20>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, true, 30>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, true, 35>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, true, 40>,
    				agree_kernel<TInput, float, NXCVariant::PLAIN, true, 65>,
    				},
    				{
    				agree_kernel<TInput, double, NXCVariant::PLAIN, true, 4>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, true, 8>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, true, 12>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, true, 16>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, true, 20>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, true, 30>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, true, 35>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, true, 40>,
    				agree_kernel<TInput, double, NXCVariant::PLAIN, true, 65>,
    				},
    			},
    			{
    				{
    				agree_kernel<TInput, float, NXCVariant::MINVAR, true, 4>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, true, 8>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, true, 12>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, true, 16>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, true, 20>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, true, 30>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, true, 35>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, true, 40>,
    				agree_kernel<TInput, float, NXCVariant::MINVAR, true, 65>,
    				},
    				{
    				agree_kernel<TInput, double, NXCVariant::MINVAR, true, 4>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, true, 8>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, true, 12>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, true, 16>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, true, 20>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, true, 30>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, true, 35>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, true, 40>,
    				agree_kernel<TInput, double, NXCVariant::MINVAR, true, 65>,
    				},
    			},
    		},
    		},
    	{
    		{
    			{
    				{
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, false, 4>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, false, 8>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, false, 12>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, false, 16>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, false, 20>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, false, 30>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, false, 35>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, false, 40>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, false, 65>,
    				},
    				{
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, false, 4>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, false, 8>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, false, 12>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, false, 16>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, false, 20>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, false, 30>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, false, 35>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, false, 40>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, false, 65>,
    				},
    			},
    			{
    				{
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, false, 4>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, false, 8>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, false, 12>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, false, 16>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, false, 20>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, false, 30>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, false, 35>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, false, 40>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, false, 65>,
    				},
    				{
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, false, 4>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, false, 8>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, false, 12>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, false, 16>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, false, 20>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, false, 30>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, false, 35>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, false, 40>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, false, 65>,
    				},
    			},
    		},
    		{
    			{
    				{
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, true, 4>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, true, 8>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, true, 12>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, true, 16>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, true, 20>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, true, 30>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, true, 35>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, true, 40>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, true, 65>,
    				},
    				{
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, true, 4>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, true, 8>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, true, 12>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, true, 16>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, true, 20>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, true, 30>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, true, 35>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, true, 40>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, true, 65>,
    				},
    			},
    			{
    				{
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, true, 4>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, true, 8>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, true, 12>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, true, 16>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, true, 20>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, true, 30>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, true, 35>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, true, 40>,
    				agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, true, 65>,
    				},
    				{
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, true, 4>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, true, 8>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, true, 12>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, true, 16>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, true, 20>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, true, 30>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, true, 35>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, true, 40>,
    				agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, true, 65>,
    				},
    			},
    		},
    		},
    };
    // [[[end]]]

    if (subpixel_step.has_value())
        init_disparity<float>(out, sz);

    agree_kernel_t agree_kernel = agree_lut[subpixel_step.has_value()][corrmap != nullptr][min_var.has_value()][precision == Precision::DOUBLE][pix_idx];
    block = max_blocksize(agree_kernel);
    grid = create_grid(block, sz);

    agree_kernel<<<grid, block, 0, mainstream>>>(bicos_disp, ptrs_dev, n_images, min_nxc.value(), min_var.value_or(0.0f) * n_images, subpixel_step.value_or(-1.0f), out, GpuMatHeader(corrmap));

    if (!subpixel_step.has_value())
        out = bicos_disp;

    // clang-format on

    assertCudaSuccess(cudaGetLastError());
}

void match(
    const std::vector<cv::cuda::GpuMat>& _stack0,
    const std::vector<cv::cuda::GpuMat>& _stack1,
    cv::cuda::GpuMat& disparity,
    Config cfg,
    cv::cuda::GpuMat* corrmap,
    cv::cuda::Stream& stream
) {
    const size_t n = _stack0.size();
    const int depth = _stack0.front().depth();
    const cv::Size sz = _stack0.front().size();

    if (n < 2)
        throw Exception("need at least two images");

    if (depth != CV_8UC1 && depth != CV_16UC1)
        throw Exception("bad input depths, only CV_8UC1 and CV_16UC1 are supported");

    // clang-format off

    int required_bits = cfg.mode == TransformMode::FULL
        ? n * n - 2 * n + 3
        : 4 * n - 7;

    switch (required_bits) {
        case 0 ... 32:
            if (depth == CV_8U)
                match_impl<uint8_t, uint32_t>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            else
                match_impl<uint16_t, uint32_t>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            break;
        case 33 ... 64:
            if (depth == CV_8U)
                match_impl<uint8_t, uint64_t>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            else
                match_impl<uint16_t, uint64_t>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            break;
#ifdef BICOS_CUDA_HAS_UINT128
        case 65 ... 128:
            if (depth == CV_8U)
                match_impl<uint8_t, uint128_t>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            else
                match_impl<uint16_t, uint128_t>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            break;
        case 129 ... 256:
#else
        case 65 ... 256:
#endif
            if (depth == CV_8U)
                match_impl<uint8_t, varuint_<256>>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            else
                match_impl<uint16_t, varuint_<256>>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            break;
            break;
        default:
            throw std::invalid_argument("input stacks too large, would require " + std::to_string(required_bits) + " bits");
    }

    // clang-format on
}

} // namespace BICOS::impl::cuda