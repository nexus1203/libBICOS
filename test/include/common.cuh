#pragma once

#include <climits>
#include <opencv2/core/cuda/common.hpp>
#include <random>
#include <iostream>
#include <format>

#include "stepbuf.hpp"

namespace BICOS::test {

int randint(int from = INT_MIN, int to = INT_MAX) {
    static thread_local std::random_device dev;
    std::uniform_int_distribution<int> dist(from, to);
    int rnum = dist(dev);
    return rnum;
}

template<typename T>
void randomize(impl::cpu::StepBuf<T>& sb) {
    static thread_local std::independent_bits_engine<std::default_random_engine, CHAR_BIT, uint8_t>
        ibe;

    T* p = sb.row(0);

    std::generate(p, p + sb.size().area(), ibe);
}

dim3 create_grid(dim3 block, cv::Size sz) {
    return dim3(
        cv::cuda::device::divUp(sz.width, block.x),
        cv::cuda::device::divUp(sz.height, block.y)
    );
}

template <typename T>
T randreal(T from, T to) {
    static thread_local std::random_device dev;
    std::uniform_real_distribution<T> dist(from, to);
    return dist(dev);
}

template <typename T>
bool equals(const cv::Mat_<T>& a, const cv::Mat_<T>& b) {
    for (int row = 0; row < a.rows; ++row) {
        for (int col = 0; col < a.cols; ++col) {
            T va = ((T*)a.ptr(row))[col],
              vb = ((T*)b.ptr(row))[col];

            if (std::isnan(va) && std::isnan(vb))
                continue;

            if (va != vb) {
                std::cerr << std::format("{} != {} at ({},{})\n", va, vb, col, row);
                return false;
            }
        }
    }

    return true;
}

template<typename T>
bool equals(const impl::cpu::StepBuf<T>& a, const impl::cpu::StepBuf<T>& b, cv::Size sz) {
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

} // namespace BICOS::test
