#pragma once

#include "common.hpp"

#include <opencv2/core/cuda.hpp>

namespace BICOS::impl::cuda {

void match(
    const std::vector<cv::cuda::GpuMat>& _stack0,
    const std::vector<cv::cuda::GpuMat>& _stack1,
    cv::cuda::GpuMat& disparity,
    Config cfg,
    cv::cuda::Stream &stream
);

}