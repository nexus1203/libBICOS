/**
 *  libBICOS: binary correspondence search on multishot stereo imagery
 *  Copyright (C) 2024-2025  Robotics Group @ Julius-Maximilian University
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

#if !defined(BICOS_CUDA) && !defined(BICOS_CPU)
    #include "config.hpp"
#endif

#include <limits>
#include <opencv2/core.hpp>
#include <optional>
#include <variant>

namespace BICOS {

using uint128_t = __uint128_t;

template<typename T>
constexpr T INVALID_DISP =
    std::numeric_limits<T>::has_quiet_NaN ? std::numeric_limits<T>::quiet_NaN()
                                          : std::numeric_limits<T>::lowest();

template<typename T>
constexpr bool is_invalid(T disparity) {
#ifndef __CUDACC__
    using std::isnan;
#endif
    if constexpr (std::is_floating_point_v<T>)
        return isnan(disparity);
    else
        return disparity == INVALID_DISP<T>;
}

#if defined(BICOS_CPU)
using Image = cv::Mat;
#elif defined(BICOS_CUDA)
using Image = cv::cuda::GpuMat;
#else
    #error "unimplemented"
#endif

enum class TransformMode { LIMITED, FULL };
#if defined(BICOS_CUDA)
enum class Precision { SINGLE, DOUBLE };
#endif

namespace Variant {
    struct NoDuplicates {};
    struct Consistency {
        int max_lr_diff = 1;
        bool no_dupes = false;
    };
} // namespace Variant

using SearchVariant = std::variant<Variant::NoDuplicates, Variant::Consistency>;

struct Config {
    std::optional<float> nxcorr_threshold = 0.5f;
    std::optional<float> subpixel_step = std::nullopt;
    std::optional<float> min_variance = std::nullopt;
    TransformMode mode = TransformMode::LIMITED;
#if defined(BICOS_CUDA)
    Precision precision = Precision::SINGLE;
#endif
    SearchVariant variant = Variant::NoDuplicates {};
};

class Exception: public std::exception {
    std::string _message;

public:
    Exception(const std::string& message);
    const char* what() const throw() override;
};

} // namespace BICOS
