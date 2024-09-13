#pragma once

#include <filesystem>
#include <opencv2/core.hpp>

#include "config.hpp"

namespace BICOS {

struct SequenceEntry {
    size_t idx;
    cv::Mat m;

    bool operator<(const SequenceEntry& rhs) const {
        return idx < rhs.idx;
    }
};

void save_disparity(const cv::Mat_<BICOS::disparity_t>& disparity, const std::string& name);

void read_sequence(
    const std::filesystem::path& image_dir,
    std::vector<SequenceEntry>& lseq,
    std::vector<SequenceEntry>& rseq,
    bool force_grayscale
);

} // namespace BICOS
