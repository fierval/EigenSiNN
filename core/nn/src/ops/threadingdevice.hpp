#pragma once

#include <thread>
using std::unique_ptr;

namespace EigenSinn {

  // Single instance of each device per application

  template <typename Device_>
  class Dispatcher {
  };

  template <>
  class Dispatcher<ThreadPoolDevice> {
  public:

    ThreadPoolDevice& get_device() {
      return thread_pool_device;
    }

    static inline unique_ptr<Dispatcher<ThreadPoolDevice>>& create() {
      if (!instance) {
        instance.reset(new Dispatcher);
      }

      return instance;
    }

  private:

    Dispatcher() :
      n_devices(std::thread::hardware_concurrency())
      , _tp(n_devices > 0 ? n_devices : 2)
      , thread_pool_device(&_tp, n_devices)
    {
      if (n_devices == 0) { n_devices = 2; }
    }

    static inline std::unique_ptr<Dispatcher<ThreadPoolDevice>> instance;
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

    static inline unique_ptr<Dispatcher<DefaultDevice>>& create() {
      if (!instance) {
        instance.reset(new Dispatcher);
      }

      return instance;
    }

    DefaultDevice& get_device() {
      return cpu_device;
    }

  private:
    static inline std::unique_ptr<Dispatcher<DefaultDevice>> instance;

    Dispatcher() = default;
    DefaultDevice cpu_device;
  };

#ifdef EIGEN_USE_GPU
  template <>
  class Dispatcher<GpuDevice> {
  public:

    GpuDevice& get_device() {
      return gpu_device;
    }

    static inline unique_ptr<Dispatcher<GpuDevice>>& create() {
      if (!instance) {
        instance.reset(new Dispatcher);
      }

      return instance;
    }

  private:
    static std::unique_ptr<Dispatcher<GpuDevice>> instance;
    Dispatcher() : stream(), gpu_device(&stream) {}

    CudaStreamDevice stream;
    GpuDevice gpu_device;

  };
#endif
}