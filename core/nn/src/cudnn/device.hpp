#pragma once
#include "common.hpp"

#include "ops/opsbase.hpp"

namespace EigenSinn {
// container for cuda resources
  struct CudnnDevice
  {
    CudnnDevice() 
      : _stream()
      , _device(&_stream) {

      checkCudnnErrors(cudnnCreate(&_cudnn_handle));
    }
    ~CudnnDevice()
    {
      checkCudnnErrors(cudnnDestroy(_cudnn_handle));
    }

    cudnnHandle_t operator()() { return _cudnn_handle; }

    // pass all the device function through to GpuDevice
    EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
      return _device.allocate(num_bytes);
    }

    EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
      _device.deallocate(buffer);
    }

    EIGEN_STRONG_INLINE void* scratchpad() const {
      return _device.scratchpad();
    }

    EIGEN_STRONG_INLINE unsigned int* semaphore() const {
      return _device.semaphore();
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
      _device.memcpy(dst, src, n);
    }

    EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const {
      _device.memcpyHostToDevice(dst, src, n);
    }

    EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const {
      _device.memcpyDeviceToHost(dst, src, n);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
      _device.memset(buffer, c, n);
    }

    EIGEN_STRONG_INLINE size_t numThreads() const {
      // FIXME
      return _device.numThreads();
    }

    EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const {
      // FIXME
      return _device.firstLevelCacheSize();
    }

    EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
      return _device.lastLevelCacheSize();
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void synchronize() const {
      _device.synchronize();
    }

    EIGEN_STRONG_INLINE int getNumCudaMultiProcessors() const {
      return _device.getNumCudaMultiProcessors();
    }
    EIGEN_STRONG_INLINE int maxCudaThreadsPerBlock() const {
      return _device.maxCudaThreadsPerBlock();
    }
    EIGEN_STRONG_INLINE int maxCudaThreadsPerMultiProcessor() const {
      return _device.maxCudaThreadsPerMultiProcessor();
    }
    EIGEN_STRONG_INLINE int sharedMemPerBlock() const {
      return _device.sharedMemPerBlock();
    }
    EIGEN_STRONG_INLINE int majorDeviceVersion() const {
      return _device.majorDeviceVersion();
    }
    EIGEN_STRONG_INLINE int minorDeviceVersion() const {
      return _device.minorDeviceVersion();
    }

    EIGEN_STRONG_INLINE int maxBlocks() const {
      return _device.maxBlocks();
    }

    inline bool ok() const {
      return _device.ok();
    }

  private:

    cudnnHandle_t  _cudnn_handle;
    CudaStreamDevice _stream;
    GpuDevice _device;
  };
} // namespace EigenSinn