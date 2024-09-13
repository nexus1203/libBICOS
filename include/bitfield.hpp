#pragma once

#ifdef __CUDACC__
    #define LOCATION __host__ __device__ __forceinline__
#else
    #define LOCATION
#endif

namespace BICOS::impl {

template<typename T>
struct Bitfield {
    unsigned int i = 0;
    T v = 0;
    LOCATION void set(bool value) {
#if defined(BICOS_DEBUG)
        if (sizeof(T) * 8 <= i)
#if defined(__CUDACC__)
            abort();
#else
            throw std::overflow_error("Bitfield overflow");
#endif
#endif
        if (value)
            v |= 1 << i;

        i++;
    }
    LOCATION T get() const {
        return v;
    }
};

} // namespace BICOS::impl
