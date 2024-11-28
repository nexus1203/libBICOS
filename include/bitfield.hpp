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

#ifdef __CUDACC__
    #define BITFIELD_LOCATION __device__ __forceinline__
#else
    #define BITFIELD_LOCATION
#endif

namespace BICOS::impl {

template<typename T>
struct Bitfield {
    unsigned int i = 0u;
    T v = T(0);

    BITFIELD_LOCATION void set(bool value) {
#ifdef BICOS_DEBUG
        if (sizeof(T) * 8 <= i)
    #ifdef __CUDACC__
            __trap();
    #else
            throw std::overflow_error("Bitfield overflow");
    #endif
#endif
        if (value)
            v |= T(1) << i;

        i++;
    }
};

} // namespace BICOS::impl
