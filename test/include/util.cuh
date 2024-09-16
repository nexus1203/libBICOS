#pragma once

#include <climits>
#include <opencv2/core/cuda/common.hpp>
#include <random>
#include <iostream>
#include <format>

#include "stepbuf.hpp"
#include "config.hpp"

namespace BICOS::test {

template<typename T>
class RegisteredPtr {
private:
    T *_phost, *_pdev;

public:
    RegisteredPtr(T* phost, size_t n = 1, bool read_only = false): _phost(phost) {
        unsigned int flags = read_only ? cudaHostRegisterReadOnly : 0;

        cudaSafeCall(cudaHostRegister(_phost, sizeof(T) * n, flags));
        cudaSafeCall(cudaHostGetDevicePointer(&_pdev, _phost, 0));
    }
    ~RegisteredPtr() {
        cudaSafeCall(cudaHostUnregister(_phost));
    }

    RegisteredPtr(const RegisteredPtr&) = delete;
    RegisteredPtr& operator=(const RegisteredPtr&) = delete;

    operator T*() {
        return _pdev;
    }
    operator const T*() {
        return _pdev;
    }

    T* operator+(int rhs) {
        return _pdev + rhs;
    }
    const T* operator+(int rhs) const {
        return _pdev + rhs;
    }
};

int randint(int from = INT_MIN, int to = INT_MAX) {
    static thread_local std::random_device dev;
    std::uniform_int_distribution<int> dist(from, to);
    int rnum = dist(dev);
    return rnum;
}

template<typename T>
void randomize(impl::cpu::StepBuf<T>& sb) {
    static thread_local std::independent_bits_engine<std::default_random_engine, CHAR_BIT, uint8_t>
        ibe;

    T* p = sb.row(0);

    std::generate(p, p + sb.size().area(), ibe);
}

dim3 create_grid(dim3 block, cv::Size sz) {
    return dim3(
        cv::cuda::device::divUp(sz.width, block.x),
        cv::cuda::device::divUp(sz.height, block.y)
    );
}

template <typename T>
T randreal(T from, T to) {
    static thread_local std::random_device dev;
    std::uniform_real_distribution<T> dist(from, to);
    return dist(dev);
}

} // namespace BICOS::test
