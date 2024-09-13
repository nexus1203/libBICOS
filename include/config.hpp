#pragma once

#include <limits>
#include <opencv2/core.hpp>
#include <optional>

namespace BICOS {

using disparity_t = float;
using uint128_t = __uint128_t;

template<typename T>
constexpr T INVALID_DISP_ =
    std::numeric_limits<T>::has_quiet_NaN ? std::numeric_limits<T>::quiet_NaN() : (T)-1;
constexpr disparity_t INVALID_DISP = INVALID_DISP_<disparity_t>;

#if defined( BICOS_CPU )
using InputImage = cv::Mat;
using OutputImage = cv::Mat_<disparity_t>;
#elif defined( BICOS_CUDA )
using InputImage = cv::cuda::GpuMat;
using OutputImage = cv::cuda::GpuMat;
#else
    #error "unimplemented"
#endif

enum class TransformMode { LIMITED, FULL };

struct Config {
    double nxcorr_thresh;
    std::optional<float> subpixel_step;
    TransformMode mode;
};

} // namespace BICOS
