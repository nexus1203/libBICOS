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

#include "common.hpp"

#include <unistd.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>

#include <cxxopts.hpp>
#include <fmt/core.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#ifdef BICOS_CUDA
    #include <opencv2/core/cuda.hpp>
#endif

#include "compat.hpp"
#include "fileutils.hpp"
#include "match.hpp"

// clang-format off

#define DELTA_MS(name) double delta_##name = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - tick).count() / 1000.0

// clang-format on

using namespace BICOS;

#define LICENSE_HEADER \
    "libBICOS  Copyright (C) 2024-2025  Robotics Group @ JMU\n" \
    "This program is free software, and you are welcome to redistribute\n" \
    "it under the conditions of the GNU LGPL-3.0-or-later license.\n" \
    "Refer to https://github.com/JMUWRobotics/libBICOS for details.\n"

int main(int argc, char const* const* argv) {
    cxxopts::Options opts(argv[0], "cli to process images with BICOS");

    // clang-format off

    opts.add_options()
        ("folder0", "First folder containing input images with numbered names.", cxxopts::value<std::string>())
        ("folder1", "Optional second folder with input images. If specified, file names need to be 0.png, 1.png... Else, folder0 needs to contain 0_left.png, 0_right.png, 1_left.png...", cxxopts::value<std::string>())
        ("t,threshold", "Minimum normalized cross corellation for a match to be accepted. Set to 0.0 to disable.", cxxopts::value<float>()->default_value("0.75"))
        ("v,variance", "Minimum intensity variance. Only active with --threshold.", cxxopts::value<float>()->default_value("1.0"))
        ("s,step", "Stepsize for subpixel interpolation. Only effective when threshold is set.", cxxopts::value<float>())
        ("o,out", "Output file for disparity image.", cxxopts::value<std::string>()->default_value("bicosdisp.png"))
        ("n,stacksize", "Number of images to process. Defaults to all found in the input folders.", cxxopts::value<uint>())
        ("q,qmatrix", "Path to cv::FileStorage with single matrix \"Q\" for reconstructing a pointcloud.", cxxopts::value<std::string>())
        ("allow-negative-z", "Allow for points with negative Z values in the pointcloud output. Only effective with a given qmatrix.")
        ("m,lr-maxdiff", "Maximum disparity difference between left and right image. Enabling this disables duplicate filtering.", cxxopts::value<uint>())
#ifdef BICOS_CUDA
        ("double", "Set double instead of single precision")
#endif
        ("limited", "Limit transformation mode. Allows for more images to be used.")
        ("corrmap", "Output map of normalized cross correlation values.")
        ("no-dupes", "Default BICOS variant when --lr-maxdiff is not specified. Can be set together with --lr-maxdiff to activate both.")
        ("h,help", "Display this message.");

    opts.parse_positional({"folder0", "folder1"});
    opts.positional_help("folder0 [folder1]");

    auto args = opts.parse(argc, argv);

    // clang-format on

    if (args.count("help")) {
        fmt::println("{}", opts.help());
        return 0;
    }

    fmt::println(LICENSE_HEADER);

    if (!isatty(STDOUT_FILENO))
        std::cerr << "Danger: bicos-cli does not have a stable CLI interface\n";
    if (args.count("no-dupes") && !args.count("lr-maxdiff"))
        std::cerr << "'no-dupes' is the default when 'lr-maxdiff' is not set.\n";

    std::filesystem::path folder0 = args["folder0"].as<std::string>();
    std::filesystem::path outfile = args["out"].as<std::string>();
    std::optional<std::filesystem::path> folder1 = std::nullopt;
    std::optional<std::filesystem::path> q_store = std::nullopt;

    if (args.count("qmatrix")) {
        q_store = args["qmatrix"].as<std::string>();

        if (!std::filesystem::exists(q_store.value()))
            throw std::invalid_argument(fmt::format("'{}' does not exist", q_store.value().string()));
    }

    if (args.count("folder1"))
        folder1 = args["folder1"].as<std::string>();

    std::vector<cv::Mat> lstack, rstack;
    {
        std::vector<SequenceEntry> lseq, rseq;

        read_sequence(folder0, folder1, lseq, rseq, true);
        sort_sequence_to_stack(lseq, rseq, lstack, rstack);

        if (args.count("stacksize")) {
            uint n = args["stacksize"].as<uint>();

            if (n < lstack.size()) {
                lstack.resize(n);
                rstack.resize(n);
            }
        }

        if (lstack.size() != rstack.size())
            throw std::invalid_argument(fmt::format("Left stack: {}, right stack: {} images", lstack.size(), rstack.size()));

        fmt::println("Loaded {} {}-bit images in total", lstack.size() + rstack.size(), lstack.front().elemSize1() * 8);
    }

    // clang-format off

    BICOS::Config c {
        .nxcorr_threshold = args["threshold"].as<float>(),
        .mode = TransformMode::FULL
    };
    if (c.nxcorr_threshold.value() <= 0.0)
        c.nxcorr_threshold = std::nullopt;

    bool need_corrmap = args.count("corrmap");

    if (need_corrmap && !c.nxcorr_threshold.has_value()) {
        c.nxcorr_threshold = -1.0f;
        std::cerr << "Computing with nxcorr-threshold of " << c.nxcorr_threshold.value() << " because 'corrmap' is set\n";
    }
    if (args.count("step"))
        c.subpixel_step = args["step"].as<float>();
    if (args.count("limited"))
        c.mode = TransformMode::LIMITED;
    if (args.count("variance"))
        if (auto minvar = args["variance"].as<float>(); minvar > 0.0)
            c.min_variance = minvar;
#ifdef BICOS_CUDA
    if (args.count("double"))
        c.precision = Precision::DOUBLE;
#endif
    if (args.count("lr-maxdiff")) {
        c.variant = Variant::Consistency {
            .max_lr_diff = (int)args["lr-maxdiff"].as<uint>(),
            .no_dupes = args.count("no-dupes") > 0
        };
    }

    // clang-format on

    cv::Mat disp;
    cv::Mat corrmap;

#ifdef BICOS_CUDA

    std::vector<cv::cuda::GpuMat> lstack_gpu, rstack_gpu;

    auto tick = std::chrono::high_resolution_clock::now();

    matvec_to_gpu(lstack, rstack, lstack_gpu, rstack_gpu);

    DELTA_MS(upload);

    fmt::print("Latency:\t {}ms (upload)\t", delta_upload);
    std::cout.flush();

    cv::cuda::GpuMat disp_gpu, corr_gpu;

    tick = std::chrono::high_resolution_clock::now();

    BICOS::match(lstack_gpu, rstack_gpu, disp_gpu, c, need_corrmap ? &corr_gpu : nullptr);

    DELTA_MS(match);

    fmt::print("{}ms (match)\t", delta_match);
    std::cout.flush();

    tick = std::chrono::high_resolution_clock::now();

    disp_gpu.download(disp);
    if (need_corrmap)
        corr_gpu.download(corrmap);

    DELTA_MS(download);

    fmt::println("{}ms (download)", delta_download);

#else

    auto tick = std::chrono::high_resolution_clock::now();

    BICOS::match(lstack, rstack, disp, c, need_corrmap ? &corrmap : nullptr);

    DELTA_MS(match);

    fmt::println("Latency:\t{}ms", delta_match);

#endif

    save_image(disp, outfile);
    if (need_corrmap)
        save_image(
            corrmap,
            outfile.parent_path()
                / (outfile.stem().string() + "-corrmap" + outfile.extension().string()),
            cv::COLORMAP_VIRIDIS
        );

    if (q_store.has_value()) {
        cv::Mat Q;
        cv::Mat3f points;
        bool allow_negative_z = args.count("allow-behind");
        cv::FileStorage fs(q_store.value(), cv::FileStorage::READ);

        fs["Q"] >> Q;

        fs.release();

        cv::reprojectImageTo3D(disp, points, Q, false, CV_32F);

        switch (disp.type()) {
            case CV_16SC1:
                save_pointcloud<int16_t>(points, disp, allow_negative_z, outfile);
                break;
            case CV_32FC1:
                save_pointcloud<float>(points, disp, allow_negative_z, outfile);
                break;
            default:
                throw std::runtime_error(fmt::format("Got unexpected disparity type: {}", disp.type()));
        }
    }

    return 0;
}
