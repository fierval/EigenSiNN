#pragma once

#include "unsupported/Eigen/CXX11/Tensor"

using namespace Eigen;

namespace EigenSinn {

  template<typename Scalar, typename Device_>
  class TensorAdapter {

  public:

    explicit TensorAdapter(std::vector<Index> dims, Device_& _device, Scalar *_data = nullptr)
      : dimensions(dims)
      , device(_device) {

      if (data_ == nullptr) {
        this->data_ = static_cast<Scalar*>(device.allocate(compute_total_size_bytes()));
      }
      else {
        this->data_ = _data;
      }
    }

    ~TensorAdapter() {
      if (data_ != nullptr) {
        device.deallocate(data_);
      }
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

      for (Index = 0; i < Rank; i++) {
        out[i] = dims[i];
      }
      return out;
    }

    inline Scalar* data() { return data_; }

  private:

    Index compute_total_size_bytes() {

      if (dimensions.size() != 0) {
        // since we are compiled with CUDA, not using STL
        for (size_t i = 0; i < dimensions.size(); i++) {
          total_size *= dimensions[i];
        }
      }
      return total_size * sizeof(Scalar);
    }
    const Device_& device;
    Scalar* data_ = nullptr;
    const std::vector<Index> dimensions;
    Index total_size = 1;
  };
}