#pragma once

#include <opencv2/core.hpp>

namespace BICOS {

template<typename T>
class StepBuf {
private:
    T* _ptr;
    int _step;

public:
    StepBuf(cv::Size size) {
        _step = size.width;
        _ptr = new T[size.area()];
    }
    ~StepBuf() {
        delete[] _ptr;
    }

    StepBuf(const StepBuf&) = delete;
    StepBuf& operator=(const StepBuf&) = delete;

    T* row(int i) {
        return _ptr + i * _step;
    }
    const T* row(int i) const {
        return _ptr + i * _step;
    }
};

} // namespace bicos
