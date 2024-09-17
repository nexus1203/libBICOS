#pragma once

template<typename T>
class RegisteredPtr {
private:
    T *_phost, *_pdev;

public:
    RegisteredPtr(T* phost, size_t n = 1, bool read_only = false): _phost(phost) {
        unsigned int flags = read_only ? cudaHostRegisterReadOnly : 0;

        cudaSafeCall(cudaHostRegister(_phost, sizeof(T) * n, flags));
        cudaSafeCall(cudaHostGetDevicePointer(&_pdev, _phost, 0));
    }
    ~RegisteredPtr() {
        cudaSafeCall(cudaHostUnregister(_phost));
    }

    RegisteredPtr(const RegisteredPtr&) = delete;
    RegisteredPtr& operator=(const RegisteredPtr&) = delete;

    operator T*() {
        return _pdev;
    }
    operator const T*() {
        return _pdev;
    }

    T* operator+(int rhs) {
        return _pdev + rhs;
    }
    const T* operator+(int rhs) const {
        return _pdev + rhs;
    }
};
