#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda/common.hpp>

#define assertCudaSuccess(call) \
    do { \
        cudaError_t err = (call); \
        if (cudaSuccess != err) { \
            std::cerr << "libBICOS CUDA error in " << __FILE__ << " [ " << __PRETTY_FUNCTION__ \
                      << " | L" << __LINE__ << " ]: " << cudaGetErrorString(err) << std::endl; \
            abort(); \
        } \
    } while (0)

#define create_grid(block, size) \
    dim3( \
        cv::cuda::device::divUp(size.width, block.x), \
        cv::cuda::device::divUp(size.height, block.y) \
    )

template<typename T>
class RegisteredPtr {
private:
    T *_phost, *_pdev;

public:
    RegisteredPtr(T* phost, size_t n = 1, bool read_only = false): _phost(phost) {
        unsigned int flags = read_only ? cudaHostRegisterReadOnly : 0;

        assertCudaSuccess(cudaHostRegister(_phost, sizeof(T) * n, flags));
        assertCudaSuccess(cudaHostGetDevicePointer(&_pdev, _phost, 0));
    }
    ~RegisteredPtr() {
        assertCudaSuccess(cudaHostUnregister(_phost));
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
