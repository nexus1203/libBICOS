#pragma once

#include <opencv2/core.hpp>

#include "config.hpp"

#if defined(BICOS_CUDA)
    #include <opencv2/core/cuda.hpp>
#endif

namespace BICOS {

void match(
    const std::vector<InputImage>& stack0,
    const std::vector<InputImage>& stack1,
    OutputImage& disparity,
    Config cfg
#if defined(BICOS_CUDA)
    ,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null()
#endif
);

} // namespace BICOS
