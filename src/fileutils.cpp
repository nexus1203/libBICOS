#include "fileutils.hpp"

#include <format>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace BICOS {

void save_disparity(const cv::Mat_<BICOS::disparity_t>& disparity, const std::string& name) {
    cv::Mat normalized, colorized;

    std::filesystem::path outfile = "/tmp/" + name;

    cv::normalize(disparity, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    normalized.setTo(0, disparity == -1);
    cv::applyColorMap(normalized, colorized, cv::COLORMAP_TURBO);

    if (!cv::imwrite(outfile.replace_extension("png"), colorized)) {
        std::cerr << "Could not save image!" << std::endl;
    }
}

void read_sequence(
    const std::filesystem::path& image_dir,
    std::vector<SequenceEntry>& lseq,
    std::vector<SequenceEntry>& rseq,
    bool force_grayscale
) {
    namespace fs = std::filesystem;

    for (auto const& entry: fs::directory_iterator(image_dir)) {
        const fs::path path = entry.path();
        const std::string fname = path.filename().string();
        if (std::string::npos == fname.find("_")) {
            std::cerr << "Ignoring file: " << fname << std::endl;
            continue;
        }

        auto idx = stoul(fname);
        cv::Mat m = cv::imread(path, force_grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
        if (4 == m.channels()) {
            cv::Mat no_alpha;
            cv::cvtColor(m, no_alpha, cv::COLOR_BGRA2BGR);
            m = no_alpha;
        }

        auto& sequence = std::string::npos != fname.find("_left") ? lseq : rseq;

        sequence.push_back((SequenceEntry) { idx, m });
    }

    if (lseq.size() != rseq.size()) {
        throw std::invalid_argument(
            std::format("Unequal number of images; left: {}, right: {}", lseq.size(), rseq.size())
        );
    }
}

void sort_sequence_to_stack(
    std::vector<SequenceEntry> lin,
    std::vector<SequenceEntry> rin,
    std::vector<cv::Mat>& lout,
    std::vector<cv::Mat>& rout
) {
    std::sort(lin.begin(), lin.end());
    std::sort(rin.begin(), rin.end());

    lout.resize(lin.size());
    rout.resize(rin.size());

    std::transform(lin.begin(), lin.end(), lout.begin(), [](const SequenceEntry& e) {
        return e.m;
    });
    std::transform(rin.begin(), rin.end(), rout.begin(), [](const SequenceEntry& e) {
        return e.m;
    });
}

void matvec_to_gpu(
    const std::vector<cv::Mat>& lin,
    const std::vector<cv::Mat>& rin,
    std::vector<cv::cuda::GpuMat>& lout,
    std::vector<cv::cuda::GpuMat>& rout
) {
    lout.resize(lin.size());
    rout.resize(rin.size());

    std::transform(lin.begin(), lin.end(), lout.begin(), [](const cv::Mat& m) { return cv::cuda::GpuMat(m); });
    std::transform(rin.begin(), rin.end(), rout.begin(), [](const cv::Mat& m) { return cv::cuda::GpuMat(m); });
}

} // namespace BICOS
