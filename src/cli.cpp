#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>

#if defined( BICOS_CUDA )
#   include <opencv2/core/cuda.hpp>
#endif

#include "match.hpp"
#include "fileutils.hpp"

using namespace BICOS;

int main(int argc, char const * const * argv) {
    std::filesystem::path folder = argc == 2 ? argv[1] : "../data";

    std::vector<cv::Mat> lstack, rstack;
    {
        std::vector<SequenceEntry> lseq, rseq;

        read_sequence(folder, lseq, rseq, true);

        std::cout << "loaded " << lseq.size() + rseq.size() << " images\n";

        sort(lseq.begin(), lseq.end());
        sort(rseq.begin(), rseq.end());

        lstack.resize(lseq.size());
        rstack.resize(rseq.size());

        std::transform(lseq.begin(), lseq.end(), lstack.begin(), [](const SequenceEntry& e) { return e.m; });
        std::transform(rseq.begin(), rseq.end(), rstack.begin(), [](const SequenceEntry& e) { return e.m; });
    }

    cv::Mat_<BICOS::disparity_t> disp;

    BICOS::Config c;
    c.nxcorr_thresh = 0.5;
    c.subpixel_step = std::nullopt;

#if defined( BICOS_CUDA )

    std::vector<cv::cuda::GpuMat> lstack_gpu, rstack_gpu;
    lstack_gpu.resize(lstack.size());
    rstack_gpu.resize(rstack.size());

    std::transform(lstack.begin(), lstack.end(), lstack_gpu.begin(), [](const cv::Mat& m) { return cv::cuda::GpuMat(m); });
    std::transform(rstack.begin(), rstack.end(), rstack_gpu.begin(), [](const cv::Mat& m) { return cv::cuda::GpuMat(m); });

    cv::cuda::GpuMat disp_gpu;

    BICOS::match(lstack_gpu, rstack_gpu, disp_gpu, c);

    disp_gpu.download(disp);

#else

    BICOS::match(lstack, rstack, disp, c);

#endif    

    save_disparity(disp, "example");
}