#include "config.hpp"
#include "cuda.hpp"

#include "bitfield.hpp"
#include "opencv2/core/hal/interface.h"
#include "stepbuf.hpp"

#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#define STACKALLOC(n, type) (type*)alloca(n * sizeof(type))

namespace BICOS::impl {

using cv::cuda::PtrStepSz;

template<typename TInput, typename TDescriptor>
static __global__ void descriptor_transform_kernel(
    const PtrStepSz<TInput>* stacks,
    size_t n,
    cv::Size size,
    StepBuf<TDescriptor>* out
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (size.width <= x || size.height <= y) {
        __syncthreads();
        return;
    }

    // caching necessary?

    extern __shared__ char timeseries[];
    TInput* pix = ((TInput*)timeseries) + n * threadIdx.x;
    Bitfield<TDescriptor> bf;

    for (size_t i = 0; i < n; ++i)
        pix[i] = stacks[i](y, x);

    __syncthreads();

    float av = 0.0f;
    for (int i = 0; i < n; ++i)
        av += pix[i];
    av /= float(n);

    // clang-format off

    int prev_pair_sums[] = { -1, -1 };
    for (size_t i = 0; i < n - 2; ++i) {
        const TInput a = pix[i + 0],
                     b = pix[i + 1],
                     c = pix[i + 2];
        
        bf.set(a < b);
        bf.set(a < c);
        bf.set(a < av);

        int& prev_pair_sum = prev_pair_sums[i % 2],
             current_sum   = a + b;

        if (-1 == prev_pair_sum) {
            prev_pair_sum = a + b;
        } else {
            bf.set(prev_pair_sum < current_sum);
            prev_pair_sum = current_sum;
        }
    }

    const TInput a = pix[n - 2],
                 b = pix[n - 1];

    bf.set(a < b);
    bf.set(a < av);
    bf.set(b < av);
    bf.set(prev_pair_sums[(n - 2) % 2] < (a + b));

    // clang-format on

    out->row(y)[x] = bf.get();
}

static __device__ __forceinline__ int ham(uint32_t a, uint32_t b) {
    return __builtin_popcount(a ^ b);
}

static __device__ __forceinline__ int ham(uint64_t a, uint64_t b) {
    return __builtin_popcountll(a ^ b);
}

static __device__ __forceinline__ int ham(uint128_t a, uint128_t b) {
    const uint128_t diff = a ^ b;
    int lo = __builtin_popcountll((uint64_t)(diff & 0xFFFFFFFFFFFFFFFFUL));
    int hi = __builtin_popcountll((uint64_t)(diff >> 64));
    return lo + hi;
}

template<typename TDescriptor>
static __global__ void bicos_kernel(
    const StepBuf<TDescriptor>* descr0,
    const StepBuf<TDescriptor>* descr1,
    PtrStepSz<int16_t> out
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out.cols <= x || out.rows <= y) {
        __syncthreads();
        return;
    }

    extern __shared__ char _row1[];
    TDescriptor* row1 = (TDescriptor*)_row1;

    const TDescriptor d0 = descr0->row(y)[x];

    for (size_t i = threadIdx.x * blockDim.x; i < out.cols && i < (threadIdx.x + 1) * blockDim.x; ++i)
        row1[i] = descr1->row(y)[i];

    __syncthreads();

    int best_col1 = -1, min_cost = INT_MAX, num_duplicate_minima = 0;

    for (size_t col1 = 0; col1 < out.cols; ++col1) {
        const TDescriptor d1 = row1[col1];

        int cost = ham(d0, d1);

        if (cost < min_cost) {
            min_cost = cost;
            best_col1 = col1;
            num_duplicate_minima = 0;
        } else if (cost == min_cost) {
            num_duplicate_minima++;
        }
    }

    if (0 < num_duplicate_minima)
        return;

    out(y, x) = abs(x - best_col1);
}

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
static __global__ void agree_kernel(
    const PtrStepSz<int16_t> raw_disp,
    const PtrStepSz<TInput>* stacks,
    size_t n,
    cv::Size sz,
    double nxcorr_threshold,
    PtrStepSz<disparity_t> out
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

    TInput *pix0 = STACKALLOC(n, TInput),
           *pix1 = STACKALLOC(n, TInput);

    for (size_t t = 0; t < n; ++t) {
        pix0[t] = stacks[t    ](y, x);
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
static __global__ void agree_subpixel_kernel(
    const PtrStepSz<int16_t> raw_disp,
    const PtrStepSz<TInput>* stacks,
    size_t n,
    cv::Size sz,
    double nxcorr_threshold,
    float subpixel_step,
    PtrStepSz<disparity_t> out
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

    TInput *pix0 = STACKALLOC(n, TInput),
           *pix1 = STACKALLOC(n, TInput);

    const PtrStepSz<TInput> *stack0 = stacks,
                            *stack1 = stacks + n;

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
        TInput *interp = STACKALLOC(n, TInput);
        float *a = STACKALLOC(n, float),
              *b = STACKALLOC(n, float),
              *c = STACKALLOC(n, float);

        for (size_t t = 0; t < n; ++t) {
            TInput y0 = stack1[t](y, idx1 - 1),
                   y1 = pix1[t],
                   y2 = stack1[t](y, idx1 + 1);

            a[t] = 0.5f * ( y0 - 2.0f * y1 + y2 );
            b[t] = 0.5f * (-y0             + y2 );
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

template<typename TInput, typename TDescriptor>
static void match_cuda_impl(
    const std::vector<cv::cuda::GpuMat>& _stack0,
    const std::vector<cv::cuda::GpuMat>& _stack1,
    size_t n_images,
    cv::Size sz,
    double nxcorr_threshold,
    std::optional<float> subpixel_step,
    cv::cuda::GpuMat& out,
    cv::cuda::Stream& _stream
) {
    std::vector<PtrStepSz<TInput>> ptrs_host(2 * n_images);
    PtrStepSz<TInput>* ptrs_dev;

    for (size_t i = 0; i < n_images; ++i) {
        ptrs_host[i] = _stack0[i];
        ptrs_host[i + n_images] = _stack1[i];
    }

    cudaSafeCall(cudaHostRegister(
        ptrs_host.data(),
        2 * n_images * sizeof(PtrStepSz<TInput>),
        cudaHostRegisterReadOnly
    ));
    cudaSafeCall(cudaHostGetDevicePointer(&ptrs_dev, ptrs_host.data(), 0));

    auto descr0 = std::make_unique<StepBuf<TDescriptor>>(sz),
         descr1 = std::make_unique<StepBuf<TDescriptor>>(sz);

    StepBuf<TDescriptor>*descr0_dev, *descr1_dev;

    cudaSafeCall(cudaHostRegister(descr0.get(), sizeof(StepBuf<TDescriptor>), 0));
    cudaSafeCall(cudaHostRegister(descr1.get(), sizeof(StepBuf<TDescriptor>), 0));
    cudaSafeCall(cudaHostGetDevicePointer(&descr0_dev, descr0.get(), 0));
    cudaSafeCall(cudaHostGetDevicePointer(&descr1_dev, descr1.get(), 0));

    size_t smem_size;
    const dim3 block(1024);
    const dim3 grid(
        cv::cuda::device::divUp(sz.width, block.x),
        cv::cuda::device::divUp(sz.height, block.y)
    );

    cudaStream_t mainstream = cv::cuda::StreamAccessor::getStream(_stream);

    /* descriptor transform */

    cudaStream_t substream0, substream1;
    cudaStreamCreate(&substream0);
    cudaStreamCreate(&substream1);

    cudaEvent_t event0, event1;
    cudaEventCreate(&event0);
    cudaEventCreate(&event1);

    smem_size = block.x * n_images * sizeof(TInput);

    descriptor_transform_kernel<TInput, TDescriptor>
        <<<grid, block, smem_size, substream0>>>(ptrs_dev, n_images, sz, descr0_dev);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaEventRecord(event0, substream0));
    descriptor_transform_kernel<TInput, TDescriptor>
        <<<grid, block, smem_size, substream1>>>(ptrs_dev + n_images, n_images, sz, descr1_dev);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaEventRecord(event1, substream1));

    cudaSafeCall(cudaStreamWaitEvent(mainstream, event0));
    cudaSafeCall(cudaStreamWaitEvent(mainstream, event1));

    /* bicos disparity */

    cv::cuda::GpuMat bicos_disp(sz, CV_16SC1);
    bicos_disp.setTo(INVALID_DISP_<int16_t>);

    smem_size = sz.width * sizeof(TDescriptor);

    bicos_kernel<TDescriptor>
        <<<grid, block, smem_size, mainstream>>>(descr0_dev, descr1_dev, bicos_disp);
    cudaSafeCall(cudaGetLastError());

    /* nxcorr */

    out.create(sz, cv::DataType<disparity_t>::type);

    // smem_size = sz.width * n_images * sizeof(TInput);

    if (subpixel_step.has_value()) {
        cudaSafeCall(cudaDeviceSetLimit(cudaLimitStackSize, 128 + 3 * n_images * ( sizeof(TInput) + sizeof(float) )));
        agree_subpixel_kernel<TInput>
            <<<grid, block, 0, mainstream>>>(bicos_disp, ptrs_dev, n_images, sz, nxcorr_threshold, subpixel_step.value(), out);
    } else {
        cudaSafeCall(cudaDeviceSetLimit(cudaLimitStackSize, 128 + 2 * n_images * sizeof(TInput)));
        agree_kernel<TInput>
            <<<grid, block, 0, mainstream>>>(bicos_disp, ptrs_dev, n_images, sz, nxcorr_threshold, out);
    }

    cudaSafeCall(cudaGetLastError());

    cudaSafeCall(cudaHostUnregister(descr1.get()));
    cudaSafeCall(cudaHostUnregister(descr0.get()));
    cudaSafeCall(cudaHostUnregister(ptrs_host.data()));
}

void match_cuda(
    const std::vector<cv::cuda::GpuMat>& _stack0,
    const std::vector<cv::cuda::GpuMat>& _stack1,
    cv::cuda::GpuMat& disparity,
    Config cfg,
    cv::cuda::Stream& stream
) {
    const size_t n_images = _stack0.size();
    const int depth = _stack0.front().depth();
    const cv::Size sz = _stack0.front().size();

    int required_bits = cfg.mode == TransformMode::FULL
        ? throw std::invalid_argument("unimplemented")
        : 4 * n_images - 7;

    switch (required_bits) {
        case 0 ... 32:
            if (depth == CV_8U)
                match_cuda_impl<uint8_t, uint32_t>(_stack0, _stack1, n_images, sz, cfg.nxcorr_thresh, cfg.subpixel_step, disparity, stream);
            else
                match_cuda_impl<uint16_t, uint32_t>(_stack0, _stack1, n_images, sz, cfg.nxcorr_thresh, cfg.subpixel_step, disparity, stream);
            break;
        case 33 ... 64:
            if (depth == CV_8U)
                match_cuda_impl<uint8_t, uint64_t>(_stack0, _stack1, n_images, sz, cfg.nxcorr_thresh, cfg.subpixel_step, disparity, stream);
            else
                match_cuda_impl<uint16_t, uint64_t>(_stack0, _stack1, n_images, sz, cfg.nxcorr_thresh, cfg.subpixel_step, disparity, stream);
            break;
        case 65 ... 128:
            if (depth == CV_8U)
                match_cuda_impl<uint8_t, uint128_t>(_stack0, _stack1, n_images, sz, cfg.nxcorr_thresh, cfg.subpixel_step, disparity, stream);
            else
                match_cuda_impl<uint16_t, uint128_t>(_stack0, _stack1, n_images, sz, cfg.nxcorr_thresh, cfg.subpixel_step, disparity, stream);
            break;
        default:
            throw std::invalid_argument("input stacks too large, exceeding 128 bits");
    }
}

} // namespace BICOS::impl