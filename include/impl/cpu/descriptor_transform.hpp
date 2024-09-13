#pragma once

#include "bitfield.hpp"
#include "config.hpp"
#include "stepbuf.hpp"

namespace BICOS::impl::cpu {

template<typename TInput, typename TDescirptor>
static TDescirptor build_descriptor(const TInput* pix, size_t n) {
    Bitfield<TDescirptor> bf;

    double av = 0.0;
    for (size_t i = 0; i < n; ++i)
        av += pix[i];
    av /= double(n);

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

    return bf.get();
}

template<typename TInput, typename TDescriptor>
std::unique_ptr<StepBuf<TDescriptor>>
descriptor_transform(const cv::Mat& s, cv::Size sz, size_t n, TransformMode m) {
    auto descriptors = std::make_unique<StepBuf<TDescriptor>>(sz);

    cv::parallel_for_(cv::Range(0, sz.height), [&](const cv::Range& range) {
        for (int row = range.start; row < range.end; ++row) {
            TDescriptor* descrow = descriptors->row(row);
            for (int col = 0; col < sz.width; ++col) {
                const TInput* pix = s.ptr<TInput>(row, col);
                descrow[col] = build_descriptor<TInput, TDescriptor>(pix, n);
            }
        }
    });

    return descriptors;
}

} // namespace BICOS::impl::cpu