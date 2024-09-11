#pragma once

#include "config.hpp"

namespace BICOS::impl {

void match_gpu(
    const std::vector<cv::cuda::GpuMat>& _stack0,
    const std::vector<cv::cuda::GpuMat>& _stack1,
    cv::cuda::GpuMat& disparity,
    Config cfg
);

}