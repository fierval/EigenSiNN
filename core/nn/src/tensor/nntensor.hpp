#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <thread>

using namespace Eigen;

namespace EigenSinn {

  enum DeviceType {
    cpu = 1,
    threadpool,
    gpu
  };

  template<typename Scalar_, int NumIndices_, DeviceType _DeviceType = cpu, int Options_ = ColMajor, typename IndexType_ = DenseIndex>
  class NnTensor : public Tensor<Scalar, NumIndices_, Options_, IndexType_> {

  public:
    using enum DeviceType device;

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE NnTensor& operator=(const OtherDerived& other)
    {
      switch (_DeviceType) {
      case cpu:
        dynamic_cast<Tensor<Scalar, NumIndices_, Options_, IndexType_>*> this->operator=(other);
      case threadpool:
        if (!inited) {
          init();
        }
        this->device(thread_pool_device) = other;
      default:
        break;
      }

      return *this;
    }

    EIGEN_STRONG_INLINE NnTensor& operator=(const NnTensor& other)
    {
      switch (_DeviceType) {
      case cpu:
        return dynamic_cast<Tensor<Scalar, NumIndices_, Options_, IndexType_>*>this->operator=(other);
      case threadpool:
        if (!inited) {
          init();
        }
        this->device(thread_pool_device) = other;
      }
      return *this;
    }


  private:
    inline void init()
    {
      inited = true;
      int i = std::thread::hardware_concurrency();
      if (i == 0) { i = 2; }
      thread_pool_device = ThreadPoolDevice(2);
    }

    ThreadPoolDevice thread_pool_device;
    bool inited = false;
  };
}
