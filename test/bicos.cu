#include "impl/cpu/bicos.hpp"
#include "impl/cuda/bicos.cuh"
#include "util.cuh"

#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <format>

using namespace BICOS;
using namespace impl;
using namespace test;

bool equals(const cv::Mat1s& a, const cv::Mat1s& b) {
    for (int row = 0; row < a.rows; ++row) {
        for (int col = 0; col < a.cols; ++col) {
            int16_t va = a.at<int16_t>(row, col),
                    vb = b.at<int16_t>(row, col);
            if (va != vb) {
                std::cerr << std::format("{} != {} at ({},{})\n", va, vb, row, col);
                return false;
            }
        }
    }

    return true;
}

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

    cudaSafeCall(cudaGetLastError());

    cv::Mat1s disp = cpu::bicos(ld, rd, randsize);

    cudaSafeCall(cudaDeviceSynchronize());

    cv::Mat1s devout;
    disp_dev.download(devout);

    if (!equals(disp, devout)) {
        return -1;
    }

    return 0;
}