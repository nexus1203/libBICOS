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

#include <bitset>
#include <opencv2/core.hpp>
#include <sstream>
#include <string>

// String formatting utilities to replace fmt library
namespace BICOS {

// Helper function to convert any type to string using stringstream
template<typename T>
std::string to_string(const T& value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

// Specialization for cv::Size
inline std::string to_string(const cv::Size& sz) {
    std::stringstream ss;
    ss << "(" << sz.height << ", " << sz.width << ")";
    return ss.str();
}

// Specialization for std::bitset
template<size_t N>
std::string to_string(const std::bitset<N>& set) {
    return set.to_string();
}

namespace Variant {
    inline std::string format_as(const Consistency& c) {
        std::stringstream ss;
        ss << "Consistency( max_lr_diff = " << c.max_lr_diff 
           << ", no_dupes = " << (c.no_dupes ? "true" : "false") << " )";
        return ss.str();
    }
    
    inline std::string format_as(const NoDuplicates&) {
        return "NoDuplicates";
    }
} // namespace Variant

#ifdef BICOS_CUDA
inline std::string format_as(const Precision& p) {
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

inline std::string format_as(const TransformMode& m) {
    switch (m) {
        case TransformMode::LIMITED:
            return "Limited";
        case TransformMode::FULL:
            return "Full";
        default:
            __builtin_unreachable();
    }
}

inline std::string format_as(const Config& c) {
    std::stringstream ss;
    ss << "Config( "
       << "nxcorr_threshold = " << (c.nxcorr_threshold ? to_string(*c.nxcorr_threshold) : "null") << ", "
       << "subpixel_step = " << (c.subpixel_step ? to_string(*c.subpixel_step) : "null") << ", "
       << "min_variance = " << (c.min_variance ? to_string(*c.min_variance) : "null") << ", "
       << "mode = " << format_as(c.mode) << ", ";
       
#ifdef BICOS_CUDA
    ss << "precision = " << format_as(c.precision) << ", ";
#endif

    if (std::holds_alternative<Variant::NoDuplicates>(c.variant)) {
        ss << "variant = " << format_as(std::get<Variant::NoDuplicates>(c.variant));
    } else if (std::holds_alternative<Variant::Consistency>(c.variant)) {
        ss << "variant = " << format_as(std::get<Variant::Consistency>(c.variant));
    }
    
    ss << " )";
    return ss.str();
}

} // namespace BICOS
