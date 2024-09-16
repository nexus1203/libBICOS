#pragma once

#ifdef __CUDACC__
    #define LOCATION __device__ __forceinline__
#else
    #define LOCATION
#endif

namespace BICOS::impl {

template<typename T>
struct Bitfield {
    unsigned int i = 0u;
    T v = T(0);

    LOCATION void set(bool value) {
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
