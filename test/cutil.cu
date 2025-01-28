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

#include "common.cuh"
#include "impl/cuda/cutil.cuh"

#include <iostream>
#include <bitset>

using namespace BICOS;
using namespace impl;

#ifdef BICOS_CUDA_HAS_UINT128

#define UINT128_MAX (((__uint128_t)UINT64_MAX << 64) | UINT64_MAX)

__global__ void load_128bit_kernel(const __uint128_t *src, __uint128_t *dst) {
    *dst = cuda::load_datacache(src);
}

int test_128(void) {
    __uint128_t x, y, *py, *px;

    cudaMalloc(&py, sizeof(__uint128_t));
    cudaMalloc(&px, sizeof(__uint128_t));

    int i;
    for (i = 0, x = 0; x < UINT128_MAX; x |= __uint128_t(UINT16_MAX) << i++ * sizeof(uint16_t)) {
        assertCudaSuccess(cudaMemcpy(px, &x, sizeof(__uint128_t), cudaMemcpyHostToDevice));
        load_128bit_kernel<<<1,1>>>(px, py);
        assertCudaSuccess(cudaMemcpy(&y, py, sizeof(__uint128_t), cudaMemcpyDeviceToHost));
        assertCudaSuccess(cudaDeviceSynchronize());

        if (x != y) {
            std::cerr << std::hex << "x: " << uint64_t(x >> 64) << ' ' << uint64_t(x) << " y: " << uint64_t(y >> 64) << ' ' << uint64_t(y) << std::endl;
            return 1; 
        }
    }

    return 0;
}

#else

int test_128(void) { return 0; }

#endif

template <size_t N>
__global__ void load_varuint_kernel(const cuda::varuint_<N> *src, cuda::varuint_<N> *dst) {
    *dst = cuda::load_datacache(src);
}

template <size_t N>
int test_varuint(void) {
    constexpr size_t sz = sizeof(cuda::varuint_<N>);
    cuda::varuint_<N> x, y, *py, *px;
    
    cudaMalloc(&px, sz);
    cudaMalloc(&py, sz);

    for (size_t i = 0; i < x.size; ++i)
        x.words[i] = static_cast<uint32_t>(rand());

    assertCudaSuccess(cudaMemcpy(px, &x, sz, cudaMemcpyHostToDevice));
    load_varuint_kernel<<<1, 1>>>(px, py);
    assertCudaSuccess(cudaMemcpy(&y, py, sz, cudaMemcpyDeviceToHost));
    assertCudaSuccess(cudaDeviceSynchronize());

    return x == y ? 0 : 1;
}

int main(void) {
    return test_128() + test_varuint<256>() + test_varuint<288>();
}
