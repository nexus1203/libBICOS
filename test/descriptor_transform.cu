#include "config.hpp"
#include "impl/cpu/descriptor_transform.hpp"
#include "impl/cuda/descriptor_transform.cuh"

#include "util.cuh"

#include "fileutils.hpp"
#include <cstdint>

using namespace BICOS;
using namespace impl;
using namespace test;

dim3 create_grid(dim3 block, cv::Size sz) {
    return dim3(
        cv::cuda::device::divUp(sz.width, block.x),
        cv::cuda::device::divUp(sz.height, block.y)
    );
}

template<typename T>
bool assert_equals(const cpu::StepBuf<T>& a, const cpu::StepBuf<T>& b, cv::Size sz) {
    for (size_t row = 0; row < sz.height; ++row) {
        for (size_t col = 0; col < sz.width; ++col) {
            T va = a.row(row)[col],
              vb = b.row(row)[col];
            assert(va == vb);
        }
    }
}

int main(int argc, char const* const* argv) {
    std::filesystem::path datadir = argv[1];

    std::vector<SequenceEntry> lseq, rseq;

    std::vector<cv::Mat> lcpu, rcpu;
    std::vector<cv::cuda::GpuMat> lgpu, rgpu;

    read_sequence(datadir, lseq, rseq, true);
    sort_sequence_to_stack(lseq, rseq, lcpu, rcpu);
    matvec_to_gpu(lcpu, rcpu, lgpu, rgpu);

    cv::Mat lstack_host, rstack_host;

    cv::merge(lcpu, lstack_host);
    cv::merge(rcpu, rstack_host);

    assert(lcpu.front().type() == CV_8U);

    auto sz = lcpu.front().size();
    auto n = lcpu.size();

    auto ldesc_cpu =
        cpu::descriptor_transform<uint8_t, uint128_t>(lstack_host, sz, n, TransformMode::LIMITED);
    auto rdesc_cpu =
        cpu::descriptor_transform<uint8_t, uint128_t>(rstack_host, sz, n, TransformMode::LIMITED);

    const dim3 block(1024);
    const dim3 grid = create_grid(block, sz);

    auto ldesc_gpu = std::make_unique<cuda::StepBuf<uint128_t>>(sz),
         rdesc_gpu = std::make_unique<cuda::StepBuf<uint128_t>>(sz);
    RegisteredPtr ldesc_dev(ldesc_gpu.get()), rdesc_dev(rdesc_gpu.get());

    std::vector<cv::cuda::PtrStepSz<uint8_t>> ptrs_host(2 * n);
    RegisteredPtr ptrs_dev(ptrs_host.data(), 2 * n, true);

    for (size_t i = 0; i < n; ++i) {
        ptrs_host[i] = lgpu[i];
        ptrs_host[i + n] = rgpu[i];
    }

    cuda::descriptor_transform_kernel<uint8_t, uint128_t>
        <<<grid, block>>>(ptrs_dev, n, sz, ldesc_dev);
    cudaSafeCall(cudaDeviceSynchronize());

    cuda::descriptor_transform_kernel<uint8_t, uint128_t>
        <<<grid, block>>>(ptrs_dev + n, n, sz, rdesc_dev);
    cudaSafeCall(cudaDeviceSynchronize());

    cpu::StepBuf<uint128_t> ldesc_gpu_host(*ldesc_gpu), rdesc_gpu_host(*rdesc_gpu);

    assert_equals(*ldesc_cpu, ldesc_gpu_host, sz);
    assert_equals(*rdesc_cpu, rdesc_gpu_host, sz);

    return 0;
}
