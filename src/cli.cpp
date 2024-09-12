#include <filesystem>
#include <iostream>
#include <format>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "match.hpp"

struct SequenceEntry {
    size_t idx;
    cv::Mat m;

    bool operator<(const SequenceEntry& rhs) const {
        return idx < rhs.idx;
    }
};

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

int main(int argc, char const * const * argv) {
    std::filesystem::path folder = argc == 2 ? argv[1] : "../data";

    std::vector<cv::Mat> lstack, rstack;
    {
        std::vector<SequenceEntry> lseq, rseq;

        read_sequence(folder, lseq, rseq, true);

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
    c.subpixel_step = 0.2;

    BICOS::match(lstack, rstack, disp, c);

    save_disparity(disp, "example");
}