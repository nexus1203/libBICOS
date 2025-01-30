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

#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>

#include "bitfield.cuh"
#include "common.hpp"

namespace BICOS::impl::cuda {

#define assertCudaSuccess(call) \
    do { \
        cudaError_t err = (call); \
        if (cudaSuccess != err) { \
            std::stringstream msg("libBICOS CUDA error in "); \
            msg << __FILE__ << " [ " << __PRETTY_FUNCTION__ << " | line " << __LINE__ \
                << " ]: " << cudaGetErrorString(err); \
            throw BICOS::Exception(msg.str()); \
        } \
    } while (0)

#define create_grid(block, size) \
    dim3( \
        cv::cuda::device::divUp(size.width, block.x), \
        cv::cuda::device::divUp(size.height, block.y) \
    )

template<class T>
inline dim3 max_blocksize(T fun, size_t smem_size = 0) {
    int _minGridSize, blockSize;
    assertCudaSuccess(cudaOccupancyMaxPotentialBlockSize(&_minGridSize, &blockSize, fun, smem_size)
    );
    return dim3(blockSize);
}

template<typename T>
class RegisteredPtr {
private:
    T *_phost, *_pdev;

public:
    RegisteredPtr(T* phost, size_t n = 1, bool read_only = false): _phost(phost) {
        static thread_local int read_only_supported = -1;
        unsigned int flags = cudaHostRegisterMapped;

        if (read_only) {
            if (read_only_supported == -1) {
                cudaDeviceGetAttribute(
                    &read_only_supported,
                    cudaDevAttrHostRegisterReadOnlySupported,
                    0
                );
            }
            if (read_only_supported == 1)
                flags |= cudaHostRegisterReadOnly;
        }

        assertCudaSuccess(cudaHostRegister(_phost, sizeof(T) * n, flags));
        assertCudaSuccess(cudaHostGetDevicePointer(&_pdev, _phost, 0));
    }
    ~RegisteredPtr() {
        cudaHostUnregister(_phost);
    }

    RegisteredPtr(const RegisteredPtr&) = delete;
    RegisteredPtr& operator=(const RegisteredPtr&) = delete;

    operator T*() {
        return _pdev;
    }
    operator const T*() const {
        return _pdev;
    }

    T* operator+(int rhs) {
        return _pdev + rhs;
    }
    const T* operator+(int rhs) const {
        return _pdev + rhs;
    }
};

template<typename T>
__device__ __forceinline__ T load_datacache(const T* p) {
    return __ldg(p);
}

#ifdef BICOS_CUDA_HAS_UINT128
template<>
__device__ __forceinline__ __uint128_t load_datacache<__uint128_t>(const __uint128_t* p) {
    auto as2 = __ldg(reinterpret_cast<const ulonglong2*>(p));
    return *reinterpret_cast<__uint128_t*>(&as2);
}
#endif

template<size_t N>
__device__ __forceinline__ varuint_<N> load_datacache(const varuint_<N>* _p) {
    varuint_<N> ret;
    auto p = reinterpret_cast<const uint32_t*>(_p);

    constexpr size_t n4 = ret.size / 4, nrest = ret.size % 4;

    if constexpr (n4 > 0) {
        auto src = reinterpret_cast<const uint4*>(p);
        auto dst = reinterpret_cast<uint4*>(ret.words);
#pragma unroll
        for (size_t i = 0; i < n4; ++i)
            dst[i] = __ldg(src + i);
    }

    constexpr size_t n2 = nrest / 2, nrest2 = nrest % 2;

    if constexpr (n2 > 0) {
        auto src = reinterpret_cast<const uint2*>(p + n4 * 4);
        auto dst = reinterpret_cast<uint2*>(ret.words + n4 * 4);
        *dst = __ldg(src);
    }

    if constexpr (nrest2)
        ret.words[ret.size - 1] = __ldg(p + ret.size - 1);

    return std::move(ret);
}

template<typename T>
__device__ __forceinline__ T load_deref(const T* p) {
    return *p;
}

template<typename T>
void init_disparity(
    cv::cuda::GpuMat& map,
    cv::Size size,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null()
) {
    map.create(size, cv::DataType<T>::type);
    map.setTo(INVALID_DISP<T>, stream);
}

class GpuMatHeader {
private:
    int rows, cols;
    size_t step;
    void* data;

public:
    GpuMatHeader();
    GpuMatHeader(const cv::cuda::GpuMat &mat);
    GpuMatHeader(cv::cuda::GpuMat* ptr);

    template<typename T>
    __device__ T* ptr(int y = 0) {
        return (T*)(((uint8_t*)data) + y * step);
    }
    template<typename T>
    __device__ const T* ptr(int y = 0) const {
        return (const T*)(((const uint8_t*)data) + y * step);
    }

    template<typename T>
    __device__ T& at(int y, int x) {
        return ptr<T>(y)[x];
    }
    template<typename T>
    __device__ const T& at(int y, int x) const {
        return ptr<T>(y)[x];
    }
};

} // namespace BICOS::impl::cuda
