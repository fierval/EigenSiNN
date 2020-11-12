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

    static inline Dispatcher<ThreadPoolDevice>& create() {
      if (instance == nullptr) {
        instance = new Dispatcher;
        instance->ref_count = 0;
      }
      instance->ref_count++;
      return *instance;
    }

    ~Dispatcher() {
      if (instance != nullptr) {
        instance->ref_count--;
        if (!instance->ref_count) {
          delete instance;
          instance = nullptr;
        }
      }
    }
  private:

    Dispatcher() :
      n_devices(std::thread::hardware_concurrency())
      , _tp(n_devices > 0 ? n_devices : 2)
      , thread_pool_device(&_tp, n_devices)
    {
      if (n_devices == 0) { n_devices = 2; }
    }

    static inline Dispatcher<ThreadPoolDevice> *instance = nullptr;
    int ref_count;
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

    static inline Dispatcher<DefaultDevice>& create() {
      if (instance == nullptr) {
        instance = new Dispatcher;
        instance->ref_count = 0;
      }
      instance->ref_count++;
      return *instance;
    }

    DefaultDevice& get_device() {
      return cpu_device;
    }

    ~Dispatcher() {
      if (instance != nullptr) {
        instance->ref_count--;
        if (!instance->ref_count) {
          delete instance;
          instance = nullptr;
        }
      }
    }

  private:
    static inline Dispatcher<DefaultDevice> *instance = nullptr;
    int ref_count;
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

    static inline Dispatcher<GpuDevice>& create() {
      if (instance == nullptr) {
        instance = new Dispatcher;
        instance->ref_count = 0;
      }
      instance->ref_count++;
      return *instance;
    }

    ~Dispatcher() {
      if (instance != nullptr) {
        instance->ref_count--;
        if (!instance->ref_count) {
          delete instance;
          instance = nullptr;
        }
      }
    }

  private:
    static inline Dispatcher<GpuDevice> *instance = nullptr;
    int ref_count;
    Dispatcher() : stream(), gpu_device(&stream) {}

    CudaStreamDevice stream;
    GpuDevice gpu_device;

  };
#endif
}