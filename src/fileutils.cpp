/**
 *  libBICOS: binary correspondence search on multishot stereo imagery
 *  Copyright (C) 2024-2025  Robotics Group @ Julius-Maximilian University
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "fileutils.hpp"
#include "common.hpp"

#include <filesystem>
#include <stdexcept>

#include <fmt/core.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace BICOS {

void save_image(const cv::Mat& image, std::filesystem::path outfile, cv::ColormapTypes cmap) {
    cv::Mat normalized, colorized;
    cv::MatExpr mask, nmask;

    if (image.type() == CV_32FC1 || image.type() == CV_64FC1) {
        mask = image != image;
        nmask = image == image;
    } else {
        mask = image == INVALID_DISP<int16_t>;
        nmask = image != INVALID_DISP<int16_t>;
    }

    cv::normalize(image, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1, nmask);
    normalized.setTo(0, mask);
    cv::applyColorMap(normalized, colorized, cmap);
    colorized.setTo(0, mask);

    if (!cv::imwrite(outfile.replace_extension("png"), colorized))
        fmt::println(stderr, "Could not save to\t{}", outfile.string());
    else
        fmt::println("Saved colorized disparity to\t\t{}", outfile.string());

    if (!cv::imwrite(outfile.replace_extension("tiff"), image))
        fmt::println(stderr, "Could not save to\t{}", outfile.string());
    else
        fmt::println(stderr, "Saved floating-point disparity to\t{}", outfile.string());
}

static void
read_single_dir(const std::filesystem::path& d, bool gray, std::vector<SequenceEntry>& vec) {
    for (auto const& e: std::filesystem::directory_iterator(d)) {
        const std::filesystem::path p = e.path();
        size_t l;
        auto idx = stoul(p.filename().string(), &l);

        if (l == 0)
            throw std::invalid_argument(
                "Expecting numbered files with names NN.png; e.g 0.png, 1.png..."
            );

        cv::Mat m = cv::imread(
            p,
            gray ? (cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH) : cv::IMREAD_UNCHANGED
        );
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
            static const std::string errmsg =
                "Expecting numbered files with names NN_{left,right}.png; e.g.: 5_left.png, 10_right.png...";

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
            fmt::format("Unequal number of images; left: {}, right: {}", lseq.size(), rseq.size())
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
