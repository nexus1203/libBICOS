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

#include "common.hpp"

#include <limits>
#include <variant>
#include <bitset>

#include "cpu.hpp"

#include <fmt/core.h>

#include "impl/cpu/agree.hpp"
#include "impl/cpu/bicos.hpp"
#include "impl/cpu/descriptor_transform.hpp"

#define STR(s) #s

namespace BICOS::impl::cpu {

template<typename TInput, typename TDescriptor>
static void match_impl(
    const cv::Mat& stack0,
    const cv::Mat& stack1,
    std::optional<float> min_nxc,
    std::optional<float> step,
    std::optional<float> min_var,
    SearchVariant variant,
    TransformMode mode,
    cv::Size sz,
    size_t n,
    cv::Mat& out,
    cv::Mat* corrmap
) {
    std::unique_ptr<StepBuf<TDescriptor>> desc0, desc1;
    switch (mode) {
        case TransformMode::FULL: {
            desc0 = descriptor_transform<TInput, TDescriptor, transform_full>(stack0, sz, n);
            desc1 = descriptor_transform<TInput, TDescriptor, transform_full>(stack1, sz, n);
        } break;
        case TransformMode::LIMITED: {
            desc0 = descriptor_transform<TInput, TDescriptor, transform_limited>(stack0, sz, n),
            desc1 = descriptor_transform<TInput, TDescriptor, transform_limited>(stack1, sz, n);
        } break;
    }

    // this is optimized for the case when no subpixel interpolation is required.
    // write directly to out, which ideally, is int16_t.
    // only allocate a new buffer if we need floating point,
    // which is in case of subpixel interpolation

    // clang-format off

    if (std::holds_alternative<Variant::Consistency>(variant)) {
        Variant::Consistency consistency = std::get<Variant::Consistency>(variant);
        if (consistency.no_dupes)
            bicos<TDescriptor, BICOSFLAGS_NODUPES | BICOSFLAGS_CONSISTENCY>(desc0, desc1, consistency.max_lr_diff, sz, out);
        else
            bicos<TDescriptor, BICOSFLAGS_CONSISTENCY>(desc0, desc1, consistency.max_lr_diff, sz, out);
    } else
        bicos<TDescriptor, BICOSFLAGS_NODUPES>(desc0, desc1, -1, sz, out);

    if (min_nxc.has_value()) {
        if (corrmap) {
            corrmap->create(sz, cv::DataType<float>::type);
            corrmap->setTo(std::numeric_limits<float>::quiet_NaN());
        }

        if (step.has_value()) {
            cv::Mat_<float> floatdisp;
            agree_subpixel<TInput>(out, stack0, stack1, n, min_nxc.value(), step.value(), min_var, floatdisp, corrmap);
            out = floatdisp;
        } else
            agree<TInput>(out, stack0, stack1, n, min_nxc.value(), min_var, corrmap);
    }

    // clang-format on
}

void match(
    const std::vector<cv::Mat>& _stack0,
    const std::vector<cv::Mat>& _stack1,
    cv::Mat& disparity,
    Config cfg,
    cv::Mat* corrmap
) {
    const size_t n = _stack0.size();
    const int depth = _stack0.front().depth();

    if (n < 2)
        throw Exception("need at least two images");

    if (depth != CV_8UC1 && depth != CV_16UC1)
        throw Exception("bad input depths, only CV_8UC1 and CV_16UC1 are supported");

    cv::Mat stack0, stack1;
    cv::merge(_stack0, stack0);
    cv::merge(_stack1, stack1);

    // clang-format off

    int required_bits = cfg.mode == TransformMode::FULL
        ? n * n - 2 * n + 3
        : 4 * n - 7;

    const cv::Size size = _stack0.front().size();
    std::optional min_var = cfg.min_variance.has_value() ? std::optional(cfg.min_variance.value() * n) : std::nullopt;

    switch (required_bits) {
        case 0 ... 32:
            if (depth == CV_8U)
                match_impl<uint8_t, uint32_t>(stack0, stack1, cfg.nxcorr_threshold, cfg.subpixel_step, min_var, cfg.variant, cfg.mode, size, n, disparity, corrmap);
            else
                match_impl<uint16_t, uint32_t>(stack0, stack1, cfg.nxcorr_threshold, cfg.subpixel_step, min_var, cfg.variant, cfg.mode, size, n, disparity, corrmap);
            break;
        case 33 ... 64:
            if (depth == CV_8U)
                match_impl<uint8_t, uint64_t>(stack0, stack1, cfg.nxcorr_threshold, cfg.subpixel_step, min_var, cfg.variant, cfg.mode, size, n, disparity, corrmap);
            else
                match_impl<uint16_t, uint64_t>(stack0, stack1, cfg.nxcorr_threshold, cfg.subpixel_step, min_var, cfg.variant, cfg.mode, size, n, disparity, corrmap);
            break;
        case 65 ... 128:
            if (depth == CV_8U)
                match_impl<uint8_t, uint128_t>(stack0, stack1, cfg.nxcorr_threshold, cfg.subpixel_step, min_var, cfg.variant, cfg.mode, size, n, disparity, corrmap);
            else
                match_impl<uint16_t, uint128_t>(stack0, stack1, cfg.nxcorr_threshold, cfg.subpixel_step, min_var, cfg.variant, cfg.mode, size, n, disparity, corrmap);
            break;
        case 129 ... 256:
            if (depth == CV_8U)
                match_impl<uint8_t, std::bitset<256>>(stack0, stack1, cfg.nxcorr_threshold, cfg.subpixel_step, min_var, cfg.variant, cfg.mode, size, n, disparity, corrmap);
            else
                match_impl<uint16_t, std::bitset<256>>(stack0, stack1, cfg.nxcorr_threshold, cfg.subpixel_step, min_var, cfg.variant, cfg.mode, size, n, disparity, corrmap);
            break;
        default:
            throw std::invalid_argument(fmt::format("input stacks too large, would require {} bits", required_bits));
    }

    // clang-format on
}

} // namespace BICOS::impl::cpu