#pragma once

#include "config.hpp"
#include "stepbuf.hpp"

#include <bit>

namespace BICOS::impl::cpu {

static int ham(uint32_t a, uint32_t b) {
    return std::popcount(a ^ b);
}

static int ham(uint64_t a, uint64_t b) {
    return std::popcount(a ^ b);
}

static int ham(uint128_t a, uint128_t b) {
    uint128_t diff = a ^ b;
    return std::popcount((uint64_t)(diff & 0xFFFFFFFFFFFFFFFFUL))
        + std::popcount((uint64_t)(diff >> 64));
}

template<typename TDescriptor>
cv::Mat1s bicos(
    const std::unique_ptr<StepBuf<TDescriptor>>& desc0,
    const std::unique_ptr<StepBuf<TDescriptor>>& desc1,
    cv::Size sz
) {
    cv::Mat1s ret(sz);
    ret.setTo(INVALID_DISP_<int16_t>);

    cv::parallel_for_(cv::Range(0, ret.rows), [&](const cv::Range& r) {
        for (int row = r.start; row < r.end; ++row) {
            const TDescriptor *drow0 = desc0->row(row), *drow1 = desc1->row(row);

            for (int col0 = 0; col0 < ret.cols; ++col0) {
                const TDescriptor d0 = drow0[col0];

                int best_col1 = -1, min_cost = INT_MAX, num_duplicate_minima = 0;

                for (int col1 = 0; col1 < ret.cols; ++col1) {
                    const TDescriptor d1 = drow1[col1];

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
                    continue;

                ret(row, col0) = std::abs(col0 - best_col1);
            }
        }
    });

    return ret;
}

} // namespace BICOS::impl::cpu
