#include "common.hpp"

#include <format>

#include "cpu.hpp"

#include "impl/cpu/agree.hpp"
#include "impl/cpu/bicos.hpp"
#include "impl/cpu/descriptor_transform.hpp"

#define STR(s) #s

namespace BICOS::impl::cpu {

void match(
    const std::vector<cv::Mat>& _stack0,
    const std::vector<cv::Mat>& _stack1,
    cv::Mat_<disparity_t>& disparity,
    Config cfg
) {
    const size_t n_images = _stack0.size();
    const int depth = _stack0.front().depth();

    if (n_images < 2)
        throw std::invalid_argument("need at least two images");

    if (depth != CV_8UC1 && depth != CV_16UC1)
        throw std::invalid_argument("bad input depths, only CV_8UC1 and CV_16UC1 are supported");

    cv::Mat stack0, stack1;
    cv::merge(_stack0, stack0);
    cv::merge(_stack1, stack1);

    int required_bits = cfg.mode == TransformMode::FULL
        ? throw std::invalid_argument("unimplemented")
        : 4 * n_images - 7;

    const cv::Size img_size = _stack0.front().size();

    cv::Mat1s raw_disp;

#define TRANSFORM_COMPUTE(matdepth, descdepth) \
    do { \
        auto desc0 = \
                 descriptor_transform<matdepth, descdepth>(stack0, img_size, n_images, cfg.mode), \
             desc1 = \
                 descriptor_transform<matdepth, descdepth>(stack1, img_size, n_images, cfg.mode); \
        raw_disp = bicos(desc0, desc1, img_size); \
    } while (0)

    switch (required_bits) {
        case 0 ... 32:
            if (depth == CV_8U)
                TRANSFORM_COMPUTE(uint8_t, uint32_t);
            else
                TRANSFORM_COMPUTE(uint16_t, uint32_t);
            break;
        case 33 ... 64:
            if (depth == CV_8U)
                TRANSFORM_COMPUTE(uint8_t, uint64_t);
            else
                TRANSFORM_COMPUTE(uint16_t, uint64_t);
            break;
        case 65 ... 128:
            if (depth == CV_8U)
                TRANSFORM_COMPUTE(uint8_t, uint128_t);
            else
                TRANSFORM_COMPUTE(uint16_t, uint128_t);
            break;
        default:
            throw std::invalid_argument(
                std::format("input stacks too large, would require {} bits", required_bits)
            );
    }

    // clang-format off

    if (cfg.subpixel_step.has_value())
        if (depth == CV_8UC1)
            agree_subpixel<uint8_t>(raw_disp, stack0, stack1, n_images, cfg.nxcorr_thresh, cfg.subpixel_step.value(), disparity);
        else
            agree_subpixel<uint16_t>(raw_disp, stack0, stack1, n_images, cfg.nxcorr_thresh, cfg.subpixel_step.value(), disparity);
    else
        if (depth == CV_8UC1)
            agree<uint8_t>(raw_disp, stack0, stack1, n_images, cfg.nxcorr_thresh, disparity);
        else
            agree<uint16_t>(raw_disp, stack0, stack1, n_images, cfg.nxcorr_thresh, disparity);

    // clang-format on
}

} // namespace BICOS::impl::cpu