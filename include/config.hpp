#pragma once

#include <optional>
#include <limits>
#include <opencv2/core.hpp>

namespace BICOS {

using disparity_t = float;
using uint128_t = __uint128_t;

constexpr disparity_t INVALID_DISP = std::numeric_limits<disparity_t>::quiet_NaN();

#if defined( BICOS_CPU )
using InputImage  = cv::Mat;
using OutputImage = cv::Mat_<disparity_t>;
#elif defined( BICOS_CUDA )
using InputImage  = cv::cuda::GpuMat;
using OutputImage = cv::cuda::GpuMat;
#else
#   error "unimplemented"
#endif

enum class TransformMode {
    LIMITED,
    FULL
};

struct Config {
    double nxcorr_thresh;
    std::optional<float> subpixel_step;
    TransformMode mode;
};

} // namespace bicos
