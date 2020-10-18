#pragma once

#include <thread>

namespace EigenSinn {

  template <typename Device_>
  class Dispatcher {};

  template <>
  class Dispatcher<ThreadPoolDevice> {
  public:

    Dispatcher() : 
      n_devices(std::thread::hardware_concurrency())
      ,_tp(n_devices > 0 ? n_devices : 2)
      , thread_pool_device(&_tp, n_devices)
    {
      if (n_devices == 0) { n_devices = 2; }
    }

    ThreadPoolDevice& get_device() {
      return thread_pool_device;
    }

  private:
    int n_devices;
    ThreadPool _tp;
    ThreadPoolDevice thread_pool_device ;

  };

  enum DeviceType {
    cpu = 1,
    threadpool
  };

  template <>
  class Dispatcher<DefaultDevice> {
  public:
    Dispatcher() = default;
    
    DefaultDevice& get_device() {
      return cpu_device;
    }

  private:
    DefaultDevice cpu_device;
  };

#ifdef EIGEN_USE_GPU
  template <>
  class Dispatcher<GpuDevice> {
  public:
    Dispatcher() : gpu_device(&stream) {}

    GpuDevice& get_device() {
      return gpu_device;
    }

  private:
    CudaStreamDevice stream;
    GpuDevice gpu_device;

  };
#endif
}