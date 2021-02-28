#pragma once

#include "unsupported/Eigen/CXX11/Tensor"
#include <thread>
using std::unique_ptr;
using namespace Eigen;

namespace EigenSinn {

  // Single instance of each device per application

  template <typename Device_>
  class Dispatcher {
  };

  template <>
  class Dispatcher<DefaultDevice> {
  public:

    static inline Dispatcher<DefaultDevice>& create() {
      std::lock_guard<std::mutex> lck(mtx);

      if (instance == nullptr) {
        instance = new Dispatcher;
        instance->ref_count = 0;
      }
      instance->ref_count++;
      return *instance;
    }

    inline void release() {
      std::lock_guard<std::mutex> lck(mtx);

      if (instance != nullptr) {
        instance->ref_count--;
        if (!instance->ref_count) {
          delete instance;
          instance = nullptr;
        }
      }
    }

    DefaultDevice& get_device() {
      return cpu_device;
    }

  private:
    static inline std::mutex mtx;

    static inline Dispatcher<DefaultDevice>* instance = nullptr;
    int ref_count;
    Dispatcher() = default;
    ~Dispatcher() { }

    DefaultDevice cpu_device;
  };

#ifdef EIGEN_USE_THREADS
  template <>
  class Dispatcher<ThreadPoolDevice> {
  public:

    ThreadPoolDevice& get_device() {
      return thread_pool_device;
    }

    static inline Dispatcher<ThreadPoolDevice>& create() {
      std::lock_guard<std::mutex> lck(mtx);

      if (instance == nullptr) {
        instance = new Dispatcher;
        instance->ref_count = 0;
      }
      instance->ref_count++;
      return *instance;
    }

    inline void release() {
      std::lock_guard<std::mutex> lck(mtx);

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
      , ref_count(0)
    {
      if (n_devices == 0) { n_devices = 2; }
    }

    ~Dispatcher() {}
    static inline std::mutex mtx;

    static inline Dispatcher<ThreadPoolDevice>* instance = nullptr;
    int ref_count;
    int n_devices;
    ThreadPool _tp;
    ThreadPoolDevice thread_pool_device;

  };
#endif

#ifdef __INTELLISENSE__
#define EIGEN_USE_GPU
#endif

#ifdef EIGEN_USE_GPU
  template <>
  class Dispatcher<GpuDevice> {
  public:

    GpuDevice& get_device() {
      return gpu_device;
    }

    static inline Dispatcher<GpuDevice>& create() {
      std::lock_guard<std::mutex> lck(mtx);

      if (instance == nullptr) {
        instance = new Dispatcher;
        instance->ref_count = 0;
      }
      instance->ref_count++;
      return *instance;
    }

    inline void release() {
      std::lock_guard<std::mutex> lck(mtx);

      if (instance != nullptr) {
        instance->ref_count--;
        if (!instance->ref_count) {
          delete instance;
          instance = nullptr;
        }
      }
    }

  private:
    static inline std::mutex mtx;
    static inline Dispatcher<GpuDevice>* instance = nullptr;
    int ref_count;
    Dispatcher() : stream(), gpu_device(&stream) { gpu_device = GpuDevice(&stream); }
    ~Dispatcher() {}

    CudaStreamDevice stream;
    GpuDevice gpu_device;

  };
#endif
}