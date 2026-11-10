// Metal host resource API. Every handle and allocation is owned by metalrt.
// These entry points expose queue/copy/event operations without an Apple SDK in
// public C++ headers; native execution is supplied by metalrt's Metal backend.
#pragma once
#include <cstddef>
#include <cstdint>

// Three-dimensional threadgroup or grid extent used by MetalRtcKernel launches.
struct MetalSize3 {
    uint32_t x, y, z;

    constexpr MetalSize3(uint32_t vx = 1, uint32_t vy = 1, uint32_t vz = 1) : x{vx}, y{vy}, z{vz} {}
};

struct MetalStreamState;
struct MetalEventState;
using metalStream_t = MetalStreamState*;
using metalEvent_t = MetalEventState*;

enum metalError {
    metalSuccess = 0,
    metalErrorInvalidValue,
    metalErrorMemoryAllocation,
    metalErrorInitializationError,
    metalErrorInvalidDevice,
    metalErrorInvalidResourceHandle,
    metalErrorNotReady,
    metalErrorNotSupported,
    metalErrorUnknown
};

using metalError_t = metalError;

enum metalMemcpyKind {
    metalMemcpyHostToHost = 0,
    metalMemcpyHostToDevice = 1,
    metalMemcpyDeviceToHost = 2,
    metalMemcpyDeviceToDevice = 3,
    metalMemcpyDefault = 4
};

inline constexpr unsigned int metalStreamDefault = 0, metalStreamNonBlocking = 1;
inline constexpr unsigned int metalEventDefault = 0, metalEventBlockingSync = 1,
                              metalEventDisableTiming = 2;
inline constexpr unsigned int metalMemAttachGlobal = 1;

// Status-returning host operations; implementations use the selected runtime device.
extern "C" {
const char* metalGetErrorString(metalError_t error);
const char* metalGetErrorName(metalError_t error);
metalError_t metalGetLastError();
metalError_t metalPeekAtLastError();
metalError_t metalGetDevice(int* device);
metalError_t metalSetDevice(int device);
metalError_t metalGetDeviceCount(int* count);
metalError_t metalDeviceSynchronize();
metalError_t metalStreamCreate(metalStream_t* stream);
metalError_t metalStreamCreateWithFlags(metalStream_t* stream, unsigned int flags);
metalError_t metalStreamDestroy(metalStream_t stream);
metalError_t metalStreamSynchronize(metalStream_t stream);
metalError_t metalStreamQuery(metalStream_t stream);
metalError_t metalMalloc(void** ptr, size_t size);
metalError_t metalMallocManaged(void** ptr, size_t size, unsigned int flags = metalMemAttachGlobal);
metalError_t metalFree(void* ptr);
metalError_t metalMemset(void* devPtr, int value, size_t count);
metalError_t metalMemsetAsync(void* devPtr, int value, size_t count,
                              metalStream_t stream = nullptr);
metalError_t metalMemcpy(void* dst, const void* src, size_t count, metalMemcpyKind kind);
// Uploads snapshot host bytes, device copies remain queued, and host readback is
// a completion boundary. A direction that conflicts with a registered GPU span fails.
metalError_t metalMemcpyAsync(void* dst, const void* src, size_t count, metalMemcpyKind kind,
                              metalStream_t stream = nullptr);
metalError_t metalEventCreate(metalEvent_t* event);
metalError_t metalEventCreateWithFlags(metalEvent_t* event, unsigned int flags);
metalError_t metalEventDestroy(metalEvent_t event);
metalError_t metalEventRecord(metalEvent_t event, metalStream_t stream = nullptr);
metalError_t metalEventSynchronize(metalEvent_t event);
metalError_t metalEventQuery(metalEvent_t event);
metalError_t metalStreamWaitEvent(metalStream_t stream, metalEvent_t event, unsigned int flags = 0);
}

// Typed allocation overloads preserve the source convenience used by the reference.
template <class T> metalError_t metalMalloc(T** ptr, size_t size) {
    return metalMalloc(reinterpret_cast<void**>(ptr), size);
}

template <class T>
metalError_t metalMallocManaged(T** ptr, size_t size, unsigned int flags = metalMemAttachGlobal) {
    return metalMallocManaged(reinterpret_cast<void**>(ptr), size, flags);
}
