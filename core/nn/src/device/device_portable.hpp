#pragma once

#include <ops/opsbase.hpp>

namespace EigenSinn {

  /// <summary>
  /// Helper struct to carry tensors between layers
  /// </summary>
  /// <typeparam name="Scalar"></typeparam>
  template<typename Scalar>
  class PortableTensor {

  public:
    PortableTensor () : dims(nullptr), scalar(nullptr) {}

    template<typename T>
    PortableTensor(T& tensor) {

      dims = static_cast<void*>(&tensor.dimensions());
      scalar = static_cast<Scalar*>(tensor->data());
    }

    void* get_dims() { return dims; }
    Scalar* data() { return scalar; }

  private:
    void* dims;
    Scalar* scalar;

  };
}