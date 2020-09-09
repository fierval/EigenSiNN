#pragma once

#include "opsbase.hpp"
#include <thread>

namespace EigenSinn {

  class ThreadPoolDeviceWrapper {
  public:

    ThreadPoolDeviceWrapper() : 
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
}