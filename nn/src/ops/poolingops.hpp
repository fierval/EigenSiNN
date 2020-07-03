#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <tuple>
#include <stdexcept>
#include "opsbase.hpp"

namespace EigenSinn {

  enum PoolType : int {
    max,
    avg
  };

  template <int Rank, class Dims>
  inline bool check_valid_params(const array<int, Rank / 2>& extents, int stride, Dims& dims) {

    if (stride <= 0) {
      return false;
    }

    if ((Rank == 2 || Rank == 4) && Rank / 2 != extents.size()) {

      return false;
    }

    for (int i = 0; i < Rank / 2; i++) {
      // we are interested in the second or second and third dimensions
      // depending on the tensor: 2d or 4d
      int tensor_dim = dims[i + 1];
      int diff = tensor_dim - extents[i];

      if (diff < 0) {
        return false;
      }

      if (stride != 1 && diff % stride != 0) {
        return false;
      }
    }

    return true;
  }



  template <typename Scalar, int Rank>
  inline Tensor<Scalar, Rank> do_max_pool(Tensor<Scalar, Rank>& t, const array<int, Rank / 2>& extents, int stride) {

    return Tensor<Scalar, Rank>();
  }

  template <typename Scalar>
  inline Tensor<Scalar, 2> do_max_pool(Tensor<Scalar, 2>& t, const array<int, 1>& extents, int stride) {
    auto dims = t.dimensions();

    if (!check_valid_params<2>(extents, stride, dims)) {

      throw std::invalid_argument("Invalid pooling dimensions");
    }

    // we get the index as well as the value so we can create a mask
    Tensor<Tuple<Index, Scalar>, 1> local_pool(dims[0]);
    Tensor <Scalar, 2> output(dims[0], (dims[1] - extents[0]) / stride + 1);
    Tensor <Index, 2> mask(output.dimensions());

    array<Index, 2> starts({ 0, 0 });
    array<Index, 2> lengths({ dims[0], extents[0] });
    array<Index, 1> reduce_dims({ 1 });

    array<Index, 2> output_starts({ 0, 0 });

    for (; starts[1] + extents[0] <= dims[1]; starts[1] +=stride, output_starts[1]++ ) {

      // get the maximums and their indices
      local_pool = t.slice(starts, lengths).reduce(reduce_dims, internal::ArgMaxTupleReducer<Tuple<Index, Scalar>>());
      // split the tuple into its arrays: value and index of the input where
      // gradient will be propagated relative to the current slice
      for (int k = 0; k < local_pool.dimension(0); k++) {

        output(k, output_starts[1]) = local_pool(k).second();
        mask(k, output_starts[1]) = local_pool(k).first();
      }
    }

    return output;

  }

  template <typename Scalar>
  inline Tensor<Scalar, 4> do_max_pool(Tensor<Scalar, 4>& t, const array<int, 2>& extents, int stride) {
    auto dims = t.dimensions();

    if (!check_valid_params<2>(extents, stride, dims)) {

      throw std::invalid_argument("Invalid pooling dimensions");
    }

    Tensor<Tuple<Index, Scalar>, 2> local_pool(dims[0], dims[3]);
    Tensor <Tuple<DenseIndex, Scalar>, 4> output(dims[0], (dims[1] - extents[0]) / stride + 1, (dims[2] - extents[1]) / stride + 1);
    Tensor<Scalar, 4> mask(output.dimensions());

    array<Index, 4> starts({ 0, 0, 0, 0 });
    array<Index, 4> lengths({ dims[0], extents[0], extents[1], dims[3] });
    array<Index, 2> reduce_dims({ 1, 2 });

    array<Index, 4> output_starts({ 0, 0, 0, 0 });

    for (; starts[1] + extents[0] <= dims[1]; starts[1] += stride, output_starts[1]++) {
      for (; starts[2] + extents[1] <= dims[2]; starts[2] += stride, output_starts[2]++) {

        // get pooling results
        local_pool = t.slice(starts, lengths).reduce(reduce_dims, internal::ArgMaxTupleReducer<Tuple<Index, Scalar>>());

        // unchain indices. TODO: unwinding them in the backward pass will be harder than 2 dims
        for (int k = 0; k < local_pool.dimension(0); k++) {
          for (int j = 0; j < local_pool.dimension(1); j++) {

            output(k, output_starts[1], output_starts[2], j)(k, 0, 0, j) = local_pool(k).second();
            mask(k, output_starts[1], output_starts[2], j) = local_pool(k).first();
          }
        }
      }
    }
    return output;
  }

  template <typename Scalar, int Dim>
  inline Tensor<Scalar, Dim> do_max_pool_backward(const Tensor<Scalar, Dim>& next_layer_grad, const Tensor<Scalar, Dim>& mask, 
    const array<Index, Dim>& output_dims, const array<Index, Dim/2>& extents, int stride) {
    return Tensor<Scalra, Dim>();
  }

  template <typename Scalar>
  inline Tensor<Scalar, 2> do_max_pool_backward(const Tensor<Scalar, 2>& next_layer_grad, const Tensor<Scalar, 2>& mask, 
    const array<Index, 2>& output_dims, const array<Index, 1>& extents, int stride) {
    return Tensor<Scalra, 2>();
  }

  template <typename Scalar>
  inline Tensor<Scalar, 4> do_max_pool_backward(const Tensor<Scalar, 4>& next_layer_grad, const Tensor<Scalar, 4>& mask, 
    const array<Index,4>& output_dims, const array<Index, 2>& extents, int stride) {
    
    array<Index, 4> dims(mask.dimension(0))
    return Tensor<Scalra, 4>();
  }


}