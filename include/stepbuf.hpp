#pragma once

#include <opencv2/core.hpp>

#if defined( BICOS_CUDA ) && defined( __CUDACC__ )
#   include <opencv2/core/cuda/common.hpp>
#endif

#ifdef __CUDACC__
#define LOCATION __host__ __device__ __forceinline__
#else
#define LOCATION
#endif

namespace BICOS::impl {

template<typename T>
class StepBuf {
private:
    T* _ptr;
    size_t _step;

public:
    StepBuf(cv::Size size) {
#if defined( BICOS_CPU )
        _step = size.width;
        _ptr = new T[size.area()];
#elif defined( BICOS_CUDA )
        cudaSafeCall(cudaMallocPitch((void**)&_ptr, &_step, size.width * sizeof(T), size.height));
        _step /= sizeof(T);
#endif
    }
    ~StepBuf() {
#if defined( BICOS_CPU )
        delete[] _ptr;
#elif defined( BICOS_CUDA )
        cudaFree(_ptr);
#endif
    }

    StepBuf(const StepBuf&) = delete;
    StepBuf& operator=(const StepBuf&) = delete;

    LOCATION T* row(int i) {
        return _ptr + i * _step;
    }
    LOCATION const T* row(int i) const {
        return _ptr + i * _step;
    }
};

} // namespace bicos
