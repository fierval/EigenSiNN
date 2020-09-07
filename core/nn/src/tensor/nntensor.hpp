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

  template<typename Scalar, int NumIndices_, DeviceType _DeviceType = cpu, int Options_ = ColMajor, typename IndexType_ = DenseIndex>
  class NnTensor : public Tensor<Scalar, NumIndices_, Options_, IndexType_> {

  public:
    template<typename... IndexTypes>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE NnTensor(Index firstDimension, IndexTypes... otherDimensions) :
      Tensor<Scalar, NumIndices_, Options_, IndexType_>(firstDimension, otherDimensions...)
      , tp(2)
      ,thread_pool_device(&tp, 2)
    {
      init();
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE NnTensor& operator=(const OtherDerived& other)
    {
      switch (_DeviceType) {
      case cpu:
        dynamic_cast<Tensor<Scalar, NumIndices_, Options_, IndexType_>*> (this)->operator=(other);
      case threadpool:
        this->device(thread_pool_device) = other;
      default:
        break;
      }

      return *dynamic_cast<NnTensor *>(this);
    }

    EIGEN_STRONG_INLINE NnTensor& operator=(const NnTensor& other)
    {
      switch (_DeviceType) {
      case cpu:
        dynamic_cast<Tensor<Scalar, NumIndices_, Options_, IndexType_>*>(this)->operator=(other);
      case threadpool:
        this->device(thread_pool_device) = other;
      }
      return *dynamic_cast<NnTensor *>(this);
    }


  private:
    inline void init()
    {
      inited = true;
      int i = std::thread::hardware_concurrency();
      if (i == 0) { return; }
      ThreadPool _tp(i);
      thread_pool_device = ThreadPoolDevice(&_tp, i);
    }

    ThreadPoolDevice thread_pool_device;
    ThreadPool tp;
    bool inited = false;
  };
}
