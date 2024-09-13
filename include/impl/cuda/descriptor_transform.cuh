#pragma once

#include "stepbuf.hpp"
#include "bitfield.hpp"
#include "../util.hpp"

namespace BICOS::impl::cuda {

template<typename TInput, typename TDescriptor>
__global__ void descriptor_transform_kernel(
    const cv::cuda::PtrStepSz<TInput>* stacks,
    size_t n,
    cv::Size size,
    StepBuf<TDescriptor>* out
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (size.width <= x || size.height <= y) {
        //__syncthreads();
        return;
    }

    // caching necessary?

    TInput *pix = STACKALLOC(n, TInput);
    //TInput* pix = ((TInput*)timeseries) + n * threadIdx.x;
    Bitfield<TDescriptor> bf;

    for (size_t i = 0; i < n; ++i)
        pix[i] = stacks[i](y, x);

    //__syncthreads();

    float av = 0.0f;
    for (size_t t = 0; t < n; ++t)
        av += pix[t];
    av /= float(n);

    // clang-format off

    int prev_pair_sums[] = { -1, -1 };
    for (size_t t = 0; t < n - 2; ++t) {
        const TInput a = pix[t + 0],
                     b = pix[t + 1],
                     c = pix[t + 2];
        
        bf.set(a < b);
        bf.set(a < c);
        bf.set(a < av);

        int& prev_pair_sum = prev_pair_sums[t % 2],
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

}