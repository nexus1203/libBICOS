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

#pragma once

#include "common.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <fmt/std.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace BICOS {

struct SequenceEntry {
    size_t idx;
    cv::Mat m;

    bool operator<(const SequenceEntry& rhs) const {
        return idx < rhs.idx;
    }
};

template <typename TDisparity>
void save_pointcloud(
    const cv::Mat3f points,
    const cv::Mat_<TDisparity>& disparity,
    bool allow_negative_z,
    std::filesystem::path outfile
) {
    if (points.size() != disparity.size())
        throw Exception("save_pointcloud: invalid sizes");

    std::ofstream xyz(outfile.replace_extension("xyz"));

    size_t n_nonfinite  = 0,
           n_negative_z = 0;

    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            if (is_invalid(disparity(row, col)))
                continue;

            cv::Vec3f point = points(row, col);

            float x = point[0], y = point[1], z = point[2];

            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                n_nonfinite++;
                continue;
            }

            if (!allow_negative_z && z < 0.0f) {
                n_negative_z++;
                continue;
            }

            xyz << x << ' ' << y << ' ' << z << '\n';
        }
    }

    xyz.flush();
    xyz.close();

    fmt::println("Saved pointcloud in ascii-format to\t{}", outfile);
    if (n_nonfinite > 0)
        fmt::println(stderr, "Skipped {} points with non-finite fp values", n_nonfinite);
    if (n_negative_z > 0)
        fmt::println(stderr, "Skipped {} points with negative Z values", n_negative_z);
}

void save_image(const cv::Mat& image, std::filesystem::path outfile, cv::ColormapTypes cmap = cv::COLORMAP_TURBO);

void read_sequence(
    std::filesystem::path image_dir0,
    std::optional<std::filesystem::path> image_dir1,
    std::vector<SequenceEntry>& lseq,
    std::vector<SequenceEntry>& rseq,
    bool force_grayscale
);

void sort_sequence_to_stack(
    std::vector<SequenceEntry> lin,
    std::vector<SequenceEntry> rin,
    std::vector<cv::Mat>& lout,
    std::vector<cv::Mat>& rout
);

void matvec_to_gpu(
    const std::vector<cv::Mat>& lin,
    const std::vector<cv::Mat>& rin,
    std::vector<cv::cuda::GpuMat>& lout,
    std::vector<cv::cuda::GpuMat>& rout
);

} // namespace BICOS
