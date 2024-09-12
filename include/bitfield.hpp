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
#if defined( BICOS_DEBUG ) && !defined( __CUDACC__ )
        if (sizeof(T) * 8 <= i)
            throw std::overflow_error("Bitfield overflow");
#endif
        v |= value << i++;
    }
    LOCATION T get() const {
        return v;
    }
};

} // namespace bicos::impl
