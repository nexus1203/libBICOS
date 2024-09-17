#pragma once

#include "common.hpp"

#include <opencv2/core.hpp>

namespace BICOS::impl::cpu {

void match(
    const std::vector<cv::Mat>& stack0,
    const std::vector<cv::Mat>& stack1,
    cv::Mat_<disparity_t>& disparity,
    Config cfg
);

}
