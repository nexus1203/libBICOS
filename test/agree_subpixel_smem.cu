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

#include "common.cuh"
#include "common.hpp"
#include "impl/cuda/agree.cuh"
#include "impl/cuda/cutil.cuh"

#include <opencv2/core/cuda.hpp>

using namespace BICOS;
using namespace impl;
using namespace test;

int main(void) {
    int n = 15;

    const cv::Size randsize(randint(512, 2048), randint(256, 1024));

    std::vector<cv::cuda::GpuMat> _devinput;
    std::vector<cv::cuda::PtrStepSz<INPUT_TYPE>> devinput;

    for (int i = 0; i < 2 * n; ++i) {
        cv::Mat_<INPUT_TYPE> randmat(randsize);
        cv::randu(randmat, 0, std::numeric_limits<INPUT_TYPE>::max());

        cv::cuda::GpuMat randmat_dev(randmat);

        _devinput.push_back(randmat_dev);
        devinput.push_back(randmat_dev);
    }

    const cuda::RegisteredPtr devptr(devinput.data(), 2 * n, true);

    cv::Mat_<int16_t> randdisp(randsize);
    cv::randu(randdisp, -1, randsize.width);

    cv::cuda::GpuMat randdisp_dev;
    randdisp_dev.upload(randdisp);

    dim3 block, grid;

    double thresh = randreal(-0.9, 0.9);
    double minvar = randreal(0.1, 10.0);

    cv::cuda::GpuMat devout_gmem(randsize, cv::DataType<float>::type),
        devout_smem(randsize, cv::DataType<float>::type);

    devout_gmem.setTo(INVALID_DISP<float>);
    devout_smem.setTo(INVALID_DISP<float>);

    size_t smem_size = randsize.width * n * sizeof(INPUT_TYPE);

    float step = 0.25f;

    {
        auto kernel = cuda::agree_subpixel_kernel<INPUT_TYPE, double, cuda::NXCVariant::MINVAR, false>;

        block = cuda::max_blocksize(kernel);
        grid = create_grid(block, randsize);

        kernel<<<grid, block>>>(randdisp_dev, devptr, n, thresh, step, minvar, devout_gmem, cuda::PtrStepSz());
        assertCudaSuccess(cudaGetLastError());
    }

    {
        auto smem_kernel = cuda::agree_subpixel_kernel_smem<INPUT_TYPE, double, cuda::NXCVariant::MINVAR, false>;

        bool smem_fits = cudaSuccess == cudaFuncSetAttribute(
            smem_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );

        if (!smem_fits)
            return EXIT_TEST_SKIP;

        block = cuda::max_blocksize(smem_kernel);
        grid = create_grid(block, randsize);

        smem_kernel<<<grid, block, smem_size>>>(randdisp_dev, devptr, n, thresh, step, minvar, devout_smem, cuda::PtrStepSz());
        assertCudaSuccess(cudaGetLastError());
    }

    cv::Mat_<float> gmem, smem;
    devout_gmem.download(gmem);
    devout_smem.download(smem);

    double err = maxerr(gmem, smem);

    fmt::println("max-err: {}", err);
    if (err > 2.0) {
        return 1;
    }

    return 0;

    return 0;
}
