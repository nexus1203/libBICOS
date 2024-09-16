#include "config.hpp"
#include "impl/cpu/descriptor_transform.hpp"
#include "impl/cuda/descriptor_transform.cuh"
#include "util.cuh"

#include <opencv2/core/cuda.hpp>
#include <format>
#include <iostream>

using namespace BICOS;
using namespace impl;
using namespace test;

#define _STR(s) #s
#define STR(s) _STR(s)

template<typename T>
bool equals(const cpu::StepBuf<T>& a, const cpu::StepBuf<T>& b, cv::Size sz) {
    for (int row = 0; row < sz.height; ++row) {
        for (int col = 0; col < sz.width; ++col) {
            T va = a.row(row)[col], vb = b.row(row)[col];
            if (va != vb) {
                std::cerr << std::format("{} != {} at ({},{})\n", va, vb, col, row);
                return false;
            }
        }
    }

    return true;
}

int main(void) {
    cv::Mat hoststack;
    std::vector<cv::Mat_<INPUT_TYPE>> rand_host;
    std::vector<cv::cuda::GpuMat> _rand_dev;
    std::vector<cv::cuda::PtrStepSz<INPUT_TYPE>> rand_dev;

    const cv::Size randsize(randint(1024, 4096), randint(512, 2048));

    std::cout << "descriptor transform on " << randsize << " " << STR(INPUT_TYPE) << " " << STR(DESCRIPTOR_TYPE) << std::endl;

    int max_bits = sizeof(DESCRIPTOR_TYPE) * 8;
    size_t n = (max_bits + 7) / 4;

    for (size_t i = 0; i < n; ++i) {
        cv::Mat_<INPUT_TYPE> randmat(randsize);
        cv::randu(randmat, 0, std::numeric_limits<INPUT_TYPE>::max());
        rand_host.push_back(randmat);

        cv::cuda::GpuMat randmat_dev(randmat);
        _rand_dev.push_back(randmat_dev);
        rand_dev.push_back(randmat_dev);
    }

    RegisteredPtr rand_devptr(rand_dev.data(), n, true);

    cuda::StepBuf<DESCRIPTOR_TYPE> gpuout(randsize);
    RegisteredPtr gpuout_devptr(&gpuout);

    const dim3 block(1024);
    const dim3 grid = create_grid(block, randsize);

    cuda::descriptor_transform_kernel<INPUT_TYPE, DESCRIPTOR_TYPE>
        <<<grid, block>>>(rand_devptr, n, randsize, gpuout_devptr);

    cudaSafeCall(cudaGetLastError());

    cv::merge(rand_host, hoststack);

    auto cpuout = cpu::descriptor_transform<INPUT_TYPE, DESCRIPTOR_TYPE>(
        hoststack,
        randsize,
        n,
        TransformMode::LIMITED
    );

    cudaSafeCall(cudaDeviceSynchronize());

    cpu::StepBuf<DESCRIPTOR_TYPE> gpuout_host(gpuout);

    if (!equals(*cpuout, gpuout_host, randsize))
        return -1;

    return 0;
}
