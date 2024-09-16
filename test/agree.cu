#include "config.hpp"
#include "impl/cpu/agree.hpp"
#include "impl/cuda/agree.cuh"
#include "opencv2/core/traits.hpp"
#include "util.cuh"

#include <format>
#include <iostream>
#include <limits>
#include <opencv2/core/cuda.hpp>

using namespace BICOS;
using namespace impl;
using namespace test;

bool equals(const cv::Mat_<disparity_t>& a, const cv::Mat_<disparity_t>& b) {
    for (int row = 0; row < a.rows; ++row) {
        for (int col = 0; col < a.cols; ++col) {
            disparity_t va = a.at<disparity_t>(row, col), vb = b.at<disparity_t>(row, col);

            if (std::isnan(va) && std::isnan(vb))
                continue;

            if (va != vb) {
                std::cerr << std::format("{} != {} at ({},{})\n", va, vb, row, col);
                return false;
            }
        }
    }

    return true;
}

int main(void) {
    int n = 10;

    const cv::Size randsize(randint(1024, 4096), randint(512, 2048));

    std::vector<cv::cuda::GpuMat> _devinput;
    std::vector<cv::Mat> _hostinput;
    std::vector<cv::cuda::PtrStepSz<INPUT_TYPE>> devinput;
    cv::Mat hinput_l, hinput_r;

    for (int i = 0; i < 2 * n; ++i) {
        if (i == n) {
            cv::merge(_hostinput, hinput_l);
            _hostinput.clear();
        }
        cv::Mat_<INPUT_TYPE> randmat(randsize);
        cv::randu(randmat, 0, std::numeric_limits<INPUT_TYPE>::max());

        _hostinput.push_back(randmat);
        cv::cuda::GpuMat randmat_dev(randmat);

        _devinput.push_back(randmat_dev);
        devinput.push_back(randmat_dev);
    }
    cv::merge(_hostinput, hinput_r);

    RegisteredPtr devptr(devinput.data(), 2 * n, true);

    cv::Mat1s randdisp(randsize);
    cv::randu(randdisp, -1, randsize.width);
    cv::cuda::GpuMat randdisp_dev(randdisp);

    const dim3 block(512);
    const dim3 grid = create_grid(block, randsize);

    double thresh = randreal(-0.9, 0.9);

    cv::cuda::GpuMat devout(randsize, cv::DataType<disparity_t>::type);
    devout.setTo(INVALID_DISP);
    cv::Mat_<disparity_t> hostout(randsize), devout_host;

#if TEST_SUBPIXEL

    float step = randreal(0.1f, 0.5f);

    cuda::agree_subpixel_kernel<INPUT_TYPE>
        <<<grid, block>>>(randdisp_dev, devptr, n, randsize, thresh, step, devout);

    cudaSafeCall(cudaGetLastError());

    cpu::agree_subpixel<INPUT_TYPE>(randdisp, hinput_l, hinput_r, n, thresh, step, hostout);

    cudaSafeCall(cudaDeviceSynchronize());

#else

    cuda::agree_kernel<INPUT_TYPE>
        <<<grid, block>>>(randdisp_dev, devptr, n, randsize, thresh, devout);

    cudaSafeCall(cudaGetLastError());

    cpu::agree<INPUT_TYPE>(randdisp, hinput_l, hinput_r, n, thresh, hostout);

    cudaSafeCall(cudaDeviceSynchronize());

#endif

    devout.download(devout_host);

    if (!equals(hostout, devout_host))
        return 1;

    return 0;
}