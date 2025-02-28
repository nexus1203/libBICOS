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

#include "common.hpp"

#include <fmt/format.h>
#include <fmt/std.h>

#include <bitset>
#include <opencv2/core.hpp>

template<>
struct fmt::formatter<cv::Size>: formatter<string_view> {
    auto format(const cv::Size& sz, format_context& ctx) const -> format_context::iterator;
};

template<size_t N>
struct fmt::formatter<std::bitset<N>>: formatter<string_view> {
    auto format(const std::bitset<N>& set, format_context& ctx) const -> format_context::iterator {
        return format_to(ctx.out(), "{}", set.to_string());
    }
};

namespace BICOS {

namespace Variant {
    inline auto format_as(const Consistency& c) {
        return fmt::format(
            "Consistency( max_lr_diff = {}, no_dupes = {} )",
            c.max_lr_diff,
            c.no_dupes
        );
    }
    constexpr auto format_as(const NoDuplicates&) {
        return "NoDuplicates";
    }
} // namespace Variant

#ifdef BICOS_CUDA
constexpr auto format_as(const Precision& p) {
    switch (p) {
        case Precision::SINGLE:
            return "Single";
        case Precision::DOUBLE:
            return "Double";
        default:
            __builtin_unreachable();
    }
}
#endif

constexpr auto format_as(const TransformMode& m) {
    switch (m) {
        case TransformMode::LIMITED:
            return "Limited";
        case TransformMode::FULL:
            return "Full";
        default:
            __builtin_unreachable();
    }
}

inline auto format_as(const Config& c) {
    constexpr static const char* format =
        "Config( "
        "nxcorr_threshold = {}, "
        "subpixel_step = {}, "
        "min_variance = {}, "
        "mode = {}, "
#ifdef BICOS_CUDA
        "precision = {}, "
#endif
        "variant = {} "
        ")";
    return fmt::format(
        format,
        c.nxcorr_threshold,
        c.subpixel_step,
        c.min_variance,
        c.mode,
#ifdef BICOS_CUDA
        c.precision,
#endif
        c.variant
    );
}

} // namespace BICOS
