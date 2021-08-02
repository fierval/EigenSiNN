#pragma once

#include "device_wrapper.hpp"
#include "device_helpers.hpp"

using namespace Eigen;

namespace EigenSinn {

  // copying this object simply increases its ownership
  template<typename Scalar, typename Device_>
  class TensorAdapter {

  public:

    explicit TensorAdapter()
      : device(dispatcher()) {}

    explicit TensorAdapter(const std::vector<Index>& dims, Scalar* _data = nullptr)
      : TensorAdapter() {

      set_dimensions(dims);
      // cannot have other
      //dimensions = std::optional<std::vector<Index>>(dims);
      if (data_ == nullptr) {
        this->data_ = static_cast<Scalar*>(device.allocate(total_size * sizeof(Scalar)));
      }
      else {
        this->data_ = _data;
      }
    }

    explicit TensorAdapter(const TensorAdapter<Scalar, Device_>& t)
      : TensorAdapter() {

      data_ = t.data_;
      dimensions = t.dimensions;
      total_size = t.total_size;
    }

    explicit TensorAdapter(const TensorAdapter<Scalar, Device_>&& t)
      : TensorAdapter() {

      data_ = t.data_;
      dimensions = t.dimensions;
      total_size = t.total_size;
    }

    ~TensorAdapter() {
      if (data_ != nullptr) {
        device.deallocate(data_);
      }
    }

    inline TensorAdapter& operator+=(TensorAdapter& t) {
      assert(total_size == t.total_size);

#ifdef __INTELLISENSE__
#define __CUDACC__
#endif

#ifdef __CUDACC__
      if (std::is_same<GpuDevice, Device_>::value) {

        static int block(BLOCK_SIZE * BLOCK_SIZE);

        dim3 grid(getGridSize(total_size, block));

        add_kernel<Scalar> << <grid, block >> > (data(), t.data(), (long)total_size);

        cudaDeviceSynchronize();
      }
      else {
#else
      for (int i = 0; i < total_size; i++) {
        data_[i] += t.data_[i];
      }
#endif
#ifdef __CUDACC__
      }
#endif
      return *this;
    }

  friend inline TensorAdapter operator+(TensorAdapter& left, TensorAdapter& right) {

    TensorAdapter acc(left.get_dims());
    acc.deep_copy(left);

    acc += right;
    return acc;
  }

  inline Scalar* data() { return data_; }
  inline Device_& get_device() { return device; }
  inline const std::vector<Index>& get_dims() { return dimensions; }

  void setConstant(Scalar val) {
    assert(data_ != nullptr);

    std::vector<Scalar> const_mem(total_size);
    std::fill_n(const_mem.begin(), total_size, val);

    device.memcpyHostToDevice(data(), const_mem.data(), total_size * sizeof(Scalar));
  }

  void setZero() {
    setConstant(0);
  }

private:

  // should be used very rarely. This actually copies the data
  inline void deep_copy(TensorAdapter& t) {
    assert(total_size == t.total_size);
    assert(data_ != nullptr);

    device.memcpy(data_, t.data_, total_size * sizeof(Scalar));
  }

  void set_dimensions(const std::vector<Index>& dims) {

    dimensions.resize(dims.size());
    total_size = 1;
    for (int i = 0; i < dims.size(); i++) {
      dimensions[i] = dims[i];
      total_size *= dims[i];
    }
  }

  static inline DeviceWrapper<Device_> dispatcher;
  Device_& device;

  Scalar* data_ = nullptr;
  std::vector<Index> dimensions;
  Index total_size = 1;
  };
}