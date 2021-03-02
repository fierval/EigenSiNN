#pragma once

#include "dispatcher.hpp"
#include "device_helpers.hpp"

using namespace Eigen;

namespace EigenSinn {

  template<typename Scalar, typename Device_>
  class TensorAdapter {

  public:

    explicit TensorAdapter() 
      : dispatcher(Dispatcher<Device_>::create())
      , device(dispatcher.get_device()) {}

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

      dispatcher.release();
    }
    // convert between dimensions and vector types
    template<Index Rank>
    static inline std::vector<Index> dims2vec(const DSizes<Index, Rank>& dims) {
      std::vector<Index> out(Rank);

      for (Index i = 0; i < Rank; i++) {
        out[i] = dims[i];
      }
      return out;
    }

    template<Index Rank>
    static inline DSizes<Index, Rank> vec2dims(const std::vector<Index>& dims) {
      DSizes<Index, Rank> out;

      for (Index i = 0; i < Rank; i++) {
        out[i] = dims[i];
      }
      return out;
    }

    inline Scalar* data() { return data_; }
    inline Device_& get_device() { return device; }
    inline const std::vector<Index>& get_dims() { return dimensions; }

    inline TensorAdapter<Scalar, Device_> clone() {

      // will allocate memory and set dimensions
      TensorAdapter<Scalar, Device_> out(dimensions);
      device.memcpy(out.data_, data_, total_size * sizeof(Scalar));
      return out;
    }


  private:

    void set_dimensions(const std::vector<Index>& dims) {

      dimensions.resize(dims.size());
      total_size = 1;
      for (int i = 0; i < dims.size(); i++) {
        dimensions[i] = dims[i];
        total_size *= dims[i];
      }
    }

    Dispatcher<Device_>& dispatcher;
    Device_& device;

    Scalar* data_ = nullptr;
    std::vector<Index> dimensions;
    Index total_size = 1;
  };
}