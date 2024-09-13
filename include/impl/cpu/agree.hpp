#pragma once

#include "config.hpp"

namespace BICOS::impl::cpu {

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

} // namespace BICOS::impl::cpu
