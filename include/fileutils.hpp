#pragma once

#include "common.hpp"

#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace BICOS {

struct SequenceEntry {
    size_t idx;
    cv::Mat m;

    bool operator<(const SequenceEntry& rhs) const {
        return idx < rhs.idx;
    }
};

void save_disparity(const cv::Mat_<BICOS::disparity_t>& disparity, std::filesystem::path outfile);

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
