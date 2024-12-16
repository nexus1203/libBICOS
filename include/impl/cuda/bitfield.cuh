/**
 *  libBICOS: binary correspondence search on multishot stereo imagery
 *  Copyright (C) 2024  Robotics Group @ Julius-Maximilian University
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

#include <cstddef>

namespace BICOS::impl::cuda {

template<size_t NBits>
struct varuint_ {
    static_assert(NBits >= 32 && NBits % 32 == 0);
    static constexpr size_t size = NBits / sizeof(uint32_t);
    uint32_t words[size] = { 0 };

    __device__ int hamming(const varuint_<NBits>& other) const {
        int distance = 0;

#pragma unroll
        for (size_t i = 0; i < size; ++i)
            distance += __popc(words[i] ^ other.words[i]);

        return distance;
    }
};

template<typename TInteger>
struct UIntBitfield {
    unsigned int i = 0u;
    TInteger v = TInteger(0);

    __device__ void set(bool value) {
#ifdef BICOS_DEBUG
        if (sizeof(TInteger) * 8 <= i)
            __trap();
#endif
        if (value)
            v |= TInteger(1) << i;

        i++;
    }
};

template<size_t NBits>
struct ArrayBitfield {
    unsigned int i = 0u;
    varuint_<NBits> v;

    __device__ void set(bool value) {
#ifdef BICOS_DEBUG
        if (NBits <= i)
            __trap();
#endif

        if (value)
            v.words[i / 32] |= uint32_t(1) << (i & 31);

        i++;
    }
};

template<typename T>
struct Bitfield {
    using type = UIntBitfield<T>;
};

template<size_t NBits>
struct Bitfield<varuint_<NBits>> {
    using type = ArrayBitfield<NBits>;
};

template<typename T>
using Bitfield_t = typename Bitfield<T>::type;

} // namespace BICOS::impl::cuda