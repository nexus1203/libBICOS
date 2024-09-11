#pragma once

namespace bicos::impl {

template<typename T>
struct Bitfield {
    unsigned int i = 0;
    T v = 0;
    void set(bool value) {
#ifdef BICOS_DEBUG
        if (sizeof(T) * 8 <= i)
            throw std::overflow_error("Bitfield overflow");
#endif
        v |= value << i++;
    }
    T get() const {
        return v;
    }
};

} // namespace bicos::impl
