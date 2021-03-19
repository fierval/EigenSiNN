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

    explicit TensorAdapter(const std::vector<Index>& dims, Scalar *_data = nullptr)
      : TensorAdapter()  {

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

    inline Scalar* data() { return data_; }
    inline Device_& get_device() { return device; }
    inline const std::vector<Index>& get_dims() { return dimensions; }

  private:

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