#pragma once

#include <opencv2/core.hpp>

#if defined(BICOS_CUDA) && defined(__CUDACC__)
    #include <opencv2/core/cuda/common.hpp>
#endif

namespace BICOS::impl {

#if defined(BICOS_CUDA) && defined(__CUDACC__)
namespace cuda {
    template<typename>
    class StepBuf;
}
#endif

namespace cpu {

    template<typename T>
    class StepBuf {
    private:
        T* _ptr;
        size_t _step;

    public:
        StepBuf(cv::Size size) {
            _step = size.width;
            _ptr = new T[size.area()];
        }
#if defined(BICOS_CUDA) && defined(__CUDACC__)
        StepBuf(const cuda::StepBuf<T>& dev): StepBuf(dev._sz) {
            cudaSafeCall(cudaMemcpy2D(
                _ptr,
                _step * sizeof(T),
                dev._ptr,
                dev._stepb,
                _step * sizeof(T),
                dev._sz.height,
                cudaMemcpyDeviceToHost
            ));
        }
#endif
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

} // namespace cpu

#if defined(BICOS_CUDA) && defined(__CUDACC__)
namespace cuda {

    template<typename T>
    class StepBuf {
    private:
        void* _ptr;
        size_t _stepb;
        cv::Size _sz;

    public:
        StepBuf(cv::Size size) {
            cudaSafeCall(cudaMallocPitch(&_ptr, &_stepb, size.width * sizeof(T), size.height));
            _sz = size;
        }
        ~StepBuf() {
            cudaFree(_ptr);
        }
        friend class cpu::StepBuf<T>;

        StepBuf(const StepBuf&) = delete;
        StepBuf& operator=(const StepBuf&) = delete;

        __device__ __forceinline__ T* row(int i) {
            return (T*)((uint8_t*)_ptr + i * _stepb);
        }
        __device__ __forceinline__ const T* row(int i) const {
            return (T*)((uint8_t*)_ptr + i * _stepb);
        }
    };

} // namespace cuda
#endif

} // namespace BICOS::impl
