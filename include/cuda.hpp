#pragma once

#include "config.hpp"

#include <opencv2/core/cuda.hpp>

namespace BICOS::impl {

void match_cuda(
    const std::vector<cv::cuda::GpuMat>& _stack0,
    const std::vector<cv::cuda::GpuMat>& _stack1,
    cv::cuda::GpuMat& disparity,
    Config cfg,
    cv::cuda::Stream &stream
);

}