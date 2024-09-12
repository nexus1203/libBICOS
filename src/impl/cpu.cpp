#include <cstdint>
#include <format>

#include "bitfield.hpp"
#include "config.hpp"
#include "cpu.hpp"
#include "stepbuf.hpp"

#define STR(s) #s

namespace BICOS::impl {

template<typename T>
static double nxcorr(const T* pix0, const T* pix1, size_t n) {
    double mean0 = 0.0, mean1 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mean0 += pix0[i];
        mean1 += pix1[i];
    }
    mean0 /= double(n);
    mean1 /= double(n);

    double n_expectancy = 0.0, sqdiffsum0 = 0.0, sqdiffsum1 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff0 = pix0[i] - mean0, diff1 = pix1[i] - mean1;

        n_expectancy += diff0 * diff1;
        sqdiffsum0 += diff0 * diff0;
        sqdiffsum1 += diff1 * diff1;
    }

    return n_expectancy / std::sqrt(sqdiffsum0 * sqdiffsum1);
}

template<typename TInput>
static void agree_cpu(
    const cv::Mat1s& raw_disp,
    const cv::Mat& stack0,
    const cv::Mat& stack1,
    size_t n_images,
    double nxcorr_threshold,
    cv::Mat_<disparity_t>& ret
) {
    auto sz = raw_disp.size();

    ret.create(sz);
    ret.setTo(INVALID_DISP);

    cv::parallel_for_(cv::Range(0, sz.height), [&](const cv::Range& r) {
        for (int row = r.start; row < r.end; ++row) {
            const cv::Mat1s raw_row = raw_disp.row(row);
            cv::Mat_<disparity_t> ret_row = ret.row(row);

            for (int col = 0; col < sz.width; ++col) {
                const int16_t d = raw_row.at<int16_t>(col);

                if (d == INVALID_DISP_<int16_t>)
                    continue;

                const int idx1 = col - d;

                if (idx1 < 0 || sz.width <= idx1)
                    continue;

                double nxc =
                    nxcorr(stack0.ptr<TInput>(row, col), stack1.ptr<TInput>(row, idx1), n_images);

                if (nxc < nxcorr_threshold)
                    continue;

                ret_row.at<disparity_t>(col) = d;
            }
        }
    });
}

template<typename TInput>
static void agree_cpu_subpixel(
    const cv::Mat1s& raw_disp,
    const cv::Mat& stack0,
    const cv::Mat& stack1,
    size_t n_images,
    double nxcorr_threshold,
    float subpixel_step,
    cv::Mat_<disparity_t>& ret
) {
    auto sz = raw_disp.size();

    ret.create(sz);
    ret.setTo(INVALID_DISP);

    cv::parallel_for_(cv::Range(0, sz.height), [&](const cv::Range& r) {
        for (int row = r.start; row < r.end; ++row) {
            const cv::Mat1s raw_row = raw_disp.row(row);
            cv::Mat_<disparity_t> ret_row = ret.row(row);

            for (int col = 0; col < sz.width; ++col) {
                const int16_t d = raw_row.at<int16_t>(col);

                if (d == INVALID_DISP_<int16_t>)
                    continue;

                const int idx1 = col - d;

                if (idx1 < 0 || sz.width <= idx1)
                    continue;

                if (idx1 == 0 || idx1 == sz.width - 1) {
                    double nxc = nxcorr(
                        stack0.ptr<TInput>(row, col),
                        stack1.ptr<TInput>(row, idx1),
                        n_images
                    );

                    if (nxc < nxcorr_threshold)
                        continue;

                    ret_row.at<disparity_t>(col) = d;
                } else {
                    // clang-format off

                    TInput *interp = (TInput*)alloca(n_images * sizeof(TInput));

                    float *a = (float*)alloca(n_images * sizeof(float)),
                          *b = (float*)alloca(n_images * sizeof(float)),
                          *c = (float*)alloca(n_images * sizeof(float));

                    const TInput *y0 = stack1.ptr<TInput>(row, idx1 - 1),
                                 *y1 = stack1.ptr<TInput>(row, idx1    ),
                                 *y2 = stack1.ptr<TInput>(row, idx1 + 1);
                    
                    for (size_t i = 0; i < n_images; ++i) {
                        a[i] = 0.5f * ( y0[i] - 2.0f * y1[i] + y2[i] );
                        b[i] = 0.5f * (-y0[i]                + y2[i] );
                        c[i] = y1[i];
                    }

                    // clang-format on

                    float best_x = 0.0f;
                    double best_nxcorr = -1.0;

                    for (float x = -1.0f; x <= 1.0f; x += subpixel_step) {
                        for (size_t i = 0; i < n_images; ++i)
                            interp[i] = TInput(a[i] * x * x + b[i] * x + c[i]);

                        double nxc = nxcorr(stack0.ptr<TInput>(row, col), interp, n_images);

                        if (best_nxcorr < nxc) {
                            best_x = x;
                            best_nxcorr = nxc;
                        }
                    }

                    if (best_nxcorr < nxcorr_threshold)
                        continue;

                    ret_row.at<disparity_t>(col) = d + best_x;
                }
            }
        }
    });
}

int ham(uint32_t a, uint32_t b) {
    return std::popcount(a ^ b);
}
int ham(uint64_t a, uint64_t b) {
    return std::popcount(a ^ b);
}
int ham(uint128_t a, uint128_t b) {
    uint128_t diff = a ^ b;
    return std::popcount((uint64_t)(diff & 0xFFFFFFFFFFFFFFFFUL))
        + std::popcount((uint64_t)(diff >> 64));
}

template<typename TDescriptor>
static cv::Mat1s bicos(
    const std::unique_ptr<StepBuf<TDescriptor>>& desc0,
    const std::unique_ptr<StepBuf<TDescriptor>>& desc1,
    cv::Size sz
) {
    cv::Mat1s ret(sz);
    ret.setTo(INVALID_DISP_<int16_t>);

    cv::parallel_for_(cv::Range(0, ret.rows), [&](const cv::Range& r) {
        for (int row = r.start; row < r.end; ++row) {
            const TDescriptor *drow0 = desc0->row(row), *drow1 = desc1->row(row);

            for (int col0 = 0; col0 < ret.cols; ++col0) {
                const TDescriptor d0 = drow0[col0];

                int best_col1 = -1, min_cost = INT_MAX, num_duplicate_minima = 0;

                for (int col1 = 0; col0 < ret.cols; ++col1) {
                    const TDescriptor d1 = drow1[col1];

                    int cost = ham(d0, d1);

                    if (cost < min_cost) {
                        min_cost = cost;
                        best_col1 = col1;
                        num_duplicate_minima = 0;
                    } else if (cost == min_cost) {
                        num_duplicate_minima++;
                    }
                }

                if (0 < num_duplicate_minima)
                    continue;

                ret(row, col0) = std::abs(col0 - best_col1);
            }
        }
    });

    return ret;
}

template<typename TInput, typename TDescirptor>
static TDescirptor build_descriptor(const TInput* pix, size_t n) {
    Bitfield<TDescirptor> bf;

    double av = 0.0;
    for (size_t i = 0; i < n; ++i)
        av += pix[i];
    av /= double(n);

    // clang-format off

    int prev_pair_sums[] = { -1, -1 };
    for (size_t i = 0; i < n - 2; ++i) {
        const TInput a = pix[i + 0],
                     b = pix[i + 1],
                     c = pix[i + 2];
        
        bf.set(a < b);
        bf.set(a < c);
        bf.set(a < av);

        int& prev_pair_sum = prev_pair_sums[i % 2],
             current_sum   = a + b;

        if (-1 == prev_pair_sum) {
            prev_pair_sum = a + b;
        } else {
            bf.set(prev_pair_sum < current_sum);
            prev_pair_sum = current_sum;
        }
    }

    const TInput a = pix[n - 2],
                 b = pix[n - 1];

    bf.set(a < b);
    bf.set(a < av);
    bf.set(b < av);
    bf.set(prev_pair_sums[(n - 2) % 2] < (a + b));

    // clang-format on

    return bf.get();
}

template<typename TInput, typename TDescriptor>
static std::unique_ptr<StepBuf<TDescriptor>>
descriptor_transform(const cv::Mat& s, cv::Size sz, size_t n, TransformMode m) {
    auto descriptors = std::make_unique<StepBuf<TDescriptor>>(sz);

    cv::parallel_for_(cv::Range(0, sz.height), [&](const cv::Range& range) {
        for (int row = range.start; row < range.end; ++row) {
            TDescriptor* descrow = descriptors->row(row);
            for (int col = 0; col < sz.width; ++col) {
                const TInput* pix = s.ptr<TInput>(row, col);
                descrow[col] = build_descriptor<TInput, TDescriptor>(pix, n);
            }
        }
    });

    return descriptors;
}

void match_cpu(
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
            agree_cpu_subpixel<uint8_t>(raw_disp, stack0, stack1, n_images, cfg.nxcorr_thresh, cfg.subpixel_step.value(), disparity);
        else
            agree_cpu_subpixel<uint16_t>(raw_disp, stack0, stack1, n_images, cfg.nxcorr_thresh, cfg.subpixel_step.value(), disparity);
    else
        if (depth == CV_8UC1)
            agree_cpu<uint8_t>(raw_disp, stack0, stack1, n_images, cfg.nxcorr_thresh, disparity);
        else
            agree_cpu<uint16_t>(raw_disp, stack0, stack1, n_images, cfg.nxcorr_thresh, disparity);

    // clang-format on
}

} // namespace bicos::impl