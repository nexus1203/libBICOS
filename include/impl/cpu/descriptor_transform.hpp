#pragma once

#include "bitfield.hpp"
#include "config.hpp"
#include "stepbuf.hpp"

namespace BICOS::impl::cpu {

template<typename TInput, typename TDescirptor>
static TDescirptor build_descriptor(const TInput* pix, size_t n) {
    Bitfield<TDescirptor> bf;

    double av = 0.0;
    for (size_t t = 0; t < n; ++t)
        av += pix[t];
    av /= double(n);

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

        if (-1 != prev_pair_sum) {
            bf.set(prev_pair_sum < current_sum);
        }

        prev_pair_sum = current_sum;
    }

    const TInput a = pix[n - 2],
                 b = pix[n - 1];

    bf.set(a < b);
    bf.set(a < av);
    bf.set(b < av);
    bf.set(prev_pair_sums[(n - 2) % 2] < (a + b));

    // clang-format on

    return bf.get();
}

template<typename TInput, typename TDescriptor>
std::unique_ptr<StepBuf<TDescriptor>>
descriptor_transform(const cv::Mat& s, cv::Size sz, size_t n, TransformMode m) {
    auto descriptors = std::make_unique<StepBuf<TDescriptor>>(sz);

    cv::parallel_for_(cv::Range(0, sz.height), [&](const cv::Range& range) {
        for (int row = range.start; row < range.end; ++row)
            for (int col = 0; col < sz.width; ++col)
                descriptors->row(row)[col] =
                    build_descriptor<TInput, TDescriptor>(s.ptr<TInput>(row, col), n);
    });

    return descriptors;
}

} // namespace BICOS::impl::cpu