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

    const cv::Size randsize(randint(1024, 4096), randint(512, 2048));

    std::vector<cv::cuda::GpuMat> _devinput;
    std::vector<cv::cuda::PtrStepSz<INPUT_TYPE>> devinput;

    for (int i = 0; i < 2 * n; ++i) {
        cv::Mat_<INPUT_TYPE> randmat(randsize);
        cv::randu(randmat, 0, std::numeric_limits<INPUT_TYPE>::max());

        cv::cuda::GpuMat randmat_dev(randmat);

        _devinput.push_back(randmat_dev);
        devinput.push_back(randmat_dev);
    }

    RegisteredPtr devptr(devinput.data(), 2 * n, true);

    cv::Mat_<int16_t> randdisp(randsize);
    cv::randu(randdisp, -1, randsize.width);

    cv::cuda::GpuMat randdisp_dev;
    randdisp_dev.upload(randdisp);

    const dim3 block(768);
    const dim3 grid = create_grid(block, randsize);

    double thresh = randreal(-0.9, 0.9);

    cv::cuda::GpuMat devout_gmem(randsize, cv::DataType<disparity_t>::type),
        devout_smem(randsize, cv::DataType<disparity_t>::type);

    devout_gmem.setTo(INVALID_DISP);
    devout_smem.setTo(INVALID_DISP);

    size_t smem_size = randsize.width * n * sizeof(INPUT_TYPE);

    float step = 0.25f;

    cuda::agree_subpixel_kernel<INPUT_TYPE>
        <<<grid, block>>>(randdisp_dev, devptr, n, thresh, step, devout_gmem);
    assertCudaSuccess(cudaGetLastError());

    assertCudaSuccess(cudaFuncSetAttribute(
        impl::cuda::agree_subpixel_kernel_smem<INPUT_TYPE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));

    cuda::agree_subpixel_kernel_smem<INPUT_TYPE>
        <<<grid, block, smem_size>>>(randdisp_dev, devptr, n, thresh, step, devout_smem);
    assertCudaSuccess(cudaGetLastError());

    cv::Mat_<disparity_t> gmem, smem;
    devout_gmem.download(gmem);
    devout_smem.download(smem);

    double err = maxerr(gmem, smem);

    std::cout << "max-err: " << err << std::endl;
    if (err > 2.0) {
        return 1;
    }

    return 0;

    return 0;
}
