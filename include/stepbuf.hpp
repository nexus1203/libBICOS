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

#pragma once

#include <opencv2/core.hpp>

#if defined(BICOS_CUDA) && defined(__CUDACC__)
    #include "impl/cuda/cutil.cuh"
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
        cv::Size _sz;

    public:
        StepBuf(cv::Size size) {
            _step = size.width;
            _sz = size;
            _ptr = new T[size.area()];
        }
#if defined(BICOS_CUDA) && defined(__CUDACC__)
        StepBuf(const cuda::StepBuf<T>& dev): StepBuf(dev._sz) {
            assertCudaSuccess(cudaMemcpy2D(
                _ptr,
                _step * sizeof(T),
                dev._ptr,
                dev._stepb,
                _step * sizeof(T),
                dev._sz.height,
                cudaMemcpyDeviceToHost
            ));
        }
        friend class cuda::StepBuf<T>;
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
        cv::Size size() const {
            return _sz;
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
            assertCudaSuccess(cudaMallocPitch(&_ptr, &_stepb, size.width * sizeof(T), size.height));
            _sz = size;
        }
        StepBuf(const cpu::StepBuf<T>& host): StepBuf(host._sz) {
            assertCudaSuccess(cudaMemcpy2D(
                _ptr,
                _stepb,
                host._ptr,
                host._step * sizeof(T),
                _sz.width * sizeof(T),
                _sz.height,
                cudaMemcpyHostToDevice
            ));
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
