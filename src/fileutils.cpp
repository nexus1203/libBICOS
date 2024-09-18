#include "fileutils.hpp"

#include <filesystem>
#include <format>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace BICOS {

void save_disparity(const cv::Mat_<BICOS::disparity_t>& disparity, std::filesystem::path outfile) {
    cv::Mat normalized, colorized;

    cv::normalize(disparity, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    normalized.setTo(0, disparity == -1);
    cv::applyColorMap(normalized, colorized, cv::COLORMAP_TURBO);

    if (!cv::imwrite(outfile.replace_extension("png"), colorized))
        std::cerr << "Could not save to\t" << outfile << std::endl;
    else
        std::cout << "Saved colorized disparity to\t\t" << outfile << std::endl;

    if (!cv::imwrite(outfile.replace_extension("tiff"), disparity))
        std::cerr << "Could not save to\t" << outfile << std::endl;
    else
        std::cout << "Saved floating-point disparity to\t" << outfile << std::endl;
}

static void read_single_dir(auto d, bool gray, auto& vec) {
    for (auto const& e: std::filesystem::directory_iterator(d)) {
        const std::filesystem::path p = e.path();
        size_t l;
        auto idx = stoul(p.filename().string(), &l);

        if (l == 0)
            throw std::invalid_argument("Expecting numbered files with names NN.png; e.g 0.png, 1.png...");

        cv::Mat m = cv::imread(p, gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
        if (4 == m.channels()) {
            cv::Mat no_alpha;
            cv::cvtColor(m, no_alpha, cv::COLOR_BGRA2BGR);
            m = no_alpha;
        }

        vec.push_back((SequenceEntry) { idx, m });
    }
}

void read_sequence(
    std::filesystem::path image_dir0,
    std::optional<std::filesystem::path> image_dir1,
    std::vector<SequenceEntry>& lseq,
    std::vector<SequenceEntry>& rseq,
    bool force_grayscale
) {
    namespace fs = std::filesystem;

    if (image_dir1) {
        read_single_dir(image_dir0, force_grayscale, lseq);
        read_single_dir(image_dir1.value(), force_grayscale, rseq);
    } else {
        for (auto const& entry: fs::directory_iterator(image_dir0)) {
            static const std::string errmsg = "Expecting numbered files with names NN_{left,right}.png; e.g.: 5_left.png, 10_right.png...";

            const fs::path path = entry.path();
            const std::string fname = path.filename().string();

            if (std::string::npos == fname.find("_"))
                throw std::invalid_argument(errmsg);

            size_t l;
            auto idx = stoul(fname, &l);

            if (l == 0)
                throw std::invalid_argument(errmsg);

            cv::Mat m =
                cv::imread(path, force_grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
            if (4 == m.channels()) {
                cv::Mat no_alpha;
                cv::cvtColor(m, no_alpha, cv::COLOR_BGRA2BGR);
                m = no_alpha;
            }

            auto& sequence = std::string::npos != fname.find("_left") ? lseq : rseq;

            sequence.push_back((SequenceEntry) { idx, m });
        }
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

    std::transform(lin.begin(), lin.end(), lout.begin(), [](const cv::Mat& m) {
        return cv::cuda::GpuMat(m);
    });
    std::transform(rin.begin(), rin.end(), rout.begin(), [](const cv::Mat& m) {
        return cv::cuda::GpuMat(m);
    });
}

} // namespace BICOS
