#include "config.hpp"
#include "cpu.hpp"
#include "cuda.hpp"
#include "fileutils.hpp"

#include <format>
#include <iostream>
#include <optional>

bool equals(const cv::Mat_<BICOS::disparity_t>& a, const cv::Mat_<BICOS::disparity_t>& b) {
    for (int row = 0; row < a.rows; ++row) {
        for (int col = 0; col < a.cols; ++col) {
            BICOS::disparity_t va = a.at<BICOS::disparity_t>(row, col),
                               vb = b.at<BICOS::disparity_t>(row, col);

            if (std::isnan(va) && std::isnan(vb))
                continue;

            if (va != vb) {
                std::cerr << std::format("{} != {} at ({},{})\n", va, vb, col, row);
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char const* const* argv) {
    std::vector<BICOS::SequenceEntry> lseq, rseq;
    std::vector<cv::Mat> lhost, rhost;
    std::vector<cv::cuda::GpuMat> ldev, rdev;

    BICOS::read_sequence(argv[1], lseq, rseq, true);
    BICOS::sort_sequence_to_stack(lseq, rseq, lhost, rhost);
    BICOS::matvec_to_gpu(lhost, rhost, ldev, rdev);

    for (double thresh: { 0.5, 0.75, 0.9 }) {
        BICOS::Config cfg { .nxcorr_thresh = thresh,
                            .subpixel_step = std::nullopt,
                            .mode = BICOS::TransformMode::LIMITED };

        cv::Mat_<BICOS::disparity_t> dhost, ddev_host;
        cv::cuda::GpuMat ddev;

        cv::cuda::Stream stream;
        BICOS::impl::cuda::match(ldev, rdev, ddev, cfg, stream);
        ddev.download(ddev_host, stream);

        BICOS::impl::cpu::match(lhost, rhost, dhost, cfg);
        stream.waitForCompletion();

        if (!equals(dhost, ddev_host)) {
            std::cerr << "thresh: " << thresh << std::endl;
            return 1;
        }

        for (float step: { 0.1f, 0.25f, 0.5f }) {
            cfg.subpixel_step = step;

            BICOS::impl::cuda::match(ldev, rdev, ddev, cfg, stream);
            ddev.download(ddev_host, stream);

            BICOS::impl::cpu::match(lhost, rhost, dhost, cfg);
            stream.waitForCompletion();

            if (!equals(dhost, ddev_host)) {
                std::cerr << "thresh: " << thresh << " step: " << step << std::endl;
                return 1;
            }
        }
    }

    return 0;
}
