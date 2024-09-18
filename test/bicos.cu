#include "common.cuh"
#include "impl/cpu/bicos.hpp"
#include "impl/cuda/bicos.cuh"
#include "impl/cuda/cutil.cuh"

#include <opencv2/core/cuda.hpp>

using namespace BICOS;
using namespace impl;
using namespace test;

int main(void) {
    const cv::Size randsize(randint(256, 1028), randint(128, 512));

    auto ld = std::make_unique<cpu::StepBuf<DESCRIPTOR_TYPE>>(randsize),
         rd = std::make_unique<cpu::StepBuf<DESCRIPTOR_TYPE>>(randsize);

    randomize(*ld);
    randomize(*rd);

    auto ld_dev = std::make_unique<cuda::StepBuf<DESCRIPTOR_TYPE>>(*ld),
         rd_dev = std::make_unique<cuda::StepBuf<DESCRIPTOR_TYPE>>(*rd);

    RegisteredPtr lptr(ld_dev.get(), true), rptr(rd_dev.get(), true);

    cv::cuda::GpuMat disp_dev(randsize, cv::DataType<int16_t>::type);
    disp_dev.setTo(INVALID_DISP_<int16_t>);

    dim3 block(1024);
    dim3 grid = create_grid(block, randsize);

    size_t smem_size = randsize.width * sizeof(DESCRIPTOR_TYPE);
    cuda::bicos_kernel<DESCRIPTOR_TYPE><<<grid, block, smem_size>>>(lptr, rptr, disp_dev);

    assertCudaSuccess(cudaGetLastError());

    cv::Mat1s disp = cpu::bicos(ld, rd, randsize);

    assertCudaSuccess(cudaDeviceSynchronize());

    cv::Mat1s devout;
    disp_dev.download(devout);

    if (!equals(disp, devout)) {
        return 1;
    }

    return 0;
}