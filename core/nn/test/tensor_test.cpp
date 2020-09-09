#ifdef EIGEN_USE_THREADS

#include <gtest/gtest.h>
#include <ops/comparisons.hpp>
#include <ops/threadingdevice.hpp>
#include <iostream>
#include <chrono>
#include <thread>

using namespace EigenSinn;
using namespace Eigen;

namespace EigenTest {
  TEST(NnTensorTest, InitCpu) {
    
    int n_devices = std::thread::hardware_concurrency();

    if (n_devices == 0) {
      n_devices = 2;
    }

    Dispatcher<ThreadPoolDevice> dev_wrapper;
    ThreadPoolDevice& threading_device = dev_wrapper.get_device();
    DefaultDevice def_device;

    auto start = std::chrono::high_resolution_clock::now();

    Tensor<float, 2> c(3000, 3000);

    c.setConstant(2);
    Tensor<float, 2> d(3000, 3000);
    d.device(def_device) = c.sqrt();

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Took on a single cpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000. << std::endl;

    start = std::chrono::high_resolution_clock::now();
    d.device(threading_device) = c.sqrt();
    stop = std::chrono::high_resolution_clock::now();

    std::cout << "Took on a threadpool: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000. << std::endl;
  }
}
#endif