/**
 *  libBICOS: binary correspondence search on multishot stereo imagery
 *  Copyright (C) 2024  Robotics Group @ Julius-Maximilian University
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <climits>
#include <random>
#include <iostream>

#include <fmt/core.h>
#include <opencv2/core/cuda/common.hpp>

#include "stepbuf.hpp"

#define EXIT_TEST_SKIP 77 // https://mesonbuild.com/Unit-tests.html#skipped-tests-and-hard-errors

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
                fmt::println(stderr, "{} != {} at ({},{})\n", va, vb, col, row);
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
                const uint8_t *hexa = (uint8_t*)&va,
                              *hexb = (uint8_t*)&vb;

                auto f = std::cout.flags();

                std::cout << std::hex << "0x";
                for (size_t i = 0; i < sizeof(T); ++i)
                    std::cout << hexa[i];
                std::cout << " != 0x";
                for (size_t i = 0; i < sizeof(T); ++i)
                    std::cout << hexb[i];
                std::cout << " at (" << col << "," << row << ")\n";

                std::cout.flags(f);

                return false;
            }
        }
    }

    return true;
}

double maxerr(const cv::Mat_<float>& _a, const cv::Mat_<float>& _b) {
    cv::Mat a, b;
    _a.copyTo(a);
    _b.copyTo(b);

    const float magic = -10000.0f;
    cv::patchNaNs(a, magic);
    cv::patchNaNs(b, magic);

    a.setTo(0.0f, b == magic);
    b.setTo(0.0f, a == magic);
    a.setTo(0.0f, a == magic);
    b.setTo(0.0f, b == magic);

    double maxerr;

    cv::Mat absd;
    cv::absdiff(a, b, absd);
    cv::minMaxIdx(absd, NULL, &maxerr);

    return maxerr;
}

} // namespace BICOS::test
