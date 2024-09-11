#pragma once

#include <opencv2/core.hpp>

#include "config.hpp"

namespace BICOS::impl {

void match_cpu(
    const std::vector<cv::Mat>& stack0,
    const std::vector<cv::Mat>& stack1,
    cv::Mat_<disparity_t>& disparity,
    Config cfg
);

}
