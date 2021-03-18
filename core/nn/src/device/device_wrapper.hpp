#pragma once

#include "unsupported/Eigen/CXX11/Tensor"
#include <thread>
using std::unique_ptr;
using namespace Eigen;

namespace EigenSinn {

  template<typename Device_>
  class DeviceWrapper {};

  template <>
  class DeviceWrapper<DefaultDevice> {
  public:
    DeviceWrapper() {}

    const DefaultDevice& operator()() {
      return cpu_device;
    }

  private:

    DefaultDevice cpu_device;
  };

#ifdef EIGEN_USE_THREADS
  template <>
  class DeviceWrapper<ThreadPoolDevice> {
  public:
    DeviceWrapper() 
      : n_devices(std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 2)
      , _tp(n_devices)
      , thread_pool_device(&_tp, n_devices) {}


    ThreadPoolDevice& operator()() {
      return thread_pool_device;
    }

  private:
    int n_devices;
    ThreadPool _tp;
    ThreadPoolDevice thread_pool_device;

  };
#endif

#ifdef EIGEN_USE_GPU

  template<>
  class DeviceWrapper<GpuDevice> {

  public:
    DeviceWrapper()
      : stream()
      , gpu_device(&stream) {

    }

    GpuDevice& operator()() {
      return gpu_device;
    }

  private:
    CudaStreamDevice stream;
    GpuDevice gpu_device;

  };
#endif
}