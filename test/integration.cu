#include "config.hpp"
#include "cpu.hpp"
#include "cuda.hpp"
#include "fileutils.hpp"
#include <optional>

int main(int argc, char const* const* argv) {
    std::vector<BICOS::SequenceEntry> lseq, rseq;
    std::vector<cv::Mat> lhost, rhost;
    std::vector<cv::cuda::GpuMat> ldev, rdev;

    BICOS::read_sequence(argv[1], lseq, rseq, true);
    BICOS::sort_sequence_to_stack(lseq, rseq, lhost, rhost);
    BICOS::matvec_to_gpu(lhost, rhost, ldev, rdev);

    for (double thresh: { 0.5, 0.75, 0.9 }) {
        BICOS::Config cfg {
            .nxcorr_thresh = thresh,
            .subpixel_step = std::nullopt,
            .mode = BICOS::TransformMode::LIMITED
        };

        cv::Mat_<BICOS::disparity_t> dhost, ddev_host;
        cv::cuda::GpuMat ddev;

        cv::cuda::Stream stream;
        BICOS::impl::cuda::match(ldev, rdev, ddev, cfg, stream);
        ddev.download(ddev_host, stream);

        BICOS::impl::cpu::match(lhost, rhost, dhost, cfg);
        stream.waitForCompletion();

        if (!std::equal(dhost.begin(), dhost.end(), ddev_host.begin()))
            return 1;

        for (float step: { 0.1f, 0.25f, 0.5f }) {
            cfg.subpixel_step = step;

            BICOS::impl::cuda::match(ldev, rdev, ddev, cfg, stream);
            ddev.download(ddev_host, stream);

            BICOS::impl::cpu::match(lhost, rhost, dhost, cfg);
            stream.waitForCompletion();

            if (!std::equal(dhost.begin(), dhost.end(), ddev_host.begin()))
                return 1;
        }
    }

    return 0;
}
