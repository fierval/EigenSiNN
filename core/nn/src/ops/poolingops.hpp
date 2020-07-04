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
  inline auto do_max_pool(Tensor<Scalar, Rank>& t, const array<int, Rank / 2>& extents, int stride) {

    return Tuple<Tensor<Scalar, Rank>, Tensor<Index, Rank>>();
  }

  template <typename Scalar>
  inline auto do_max_pool(Tensor<Scalar, 2>& t, const array<int, 1>& extents, int stride) {
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

    for (; starts[1] + extents[0] <= dims[1]; starts[1] += stride, output_starts[1]++) {

      // get the maximums and their indices
      local_pool = t.slice(starts, lengths).reduce(reduce_dims, internal::ArgMaxTupleReducer<Tuple<Index, Scalar>>());
      // split the tuple into its arrays: value and index of the input where
      // gradient will be propagated relative to the current slice
      for (int k = 0; k < local_pool.dimension(0); k++) {

        output(k, output_starts[1]) = local_pool(k).second();
        // unroll the index into the tuple since local_pool(k).first() contains flattened index
        mask(k, output_starts[1]) = local_pool(k).first() - k * extents[0];
      }
    }

    return Tuple<Tensor<Scalar, 4>, Tensor<Index, 4>>(output, mask);

  }

  template <typename Scalar>
  inline auto do_max_pool(Tensor<Scalar, 4>& t, const array<int, 2>& extents, int stride) {
    auto dims = t.dimensions();

    if (!check_valid_params<4>(extents, stride, dims)) {

      throw std::invalid_argument("Invalid pooling dimensions");
    }

    Tensor<Tuple<Index, Scalar>, 2> local_pool(dims[0], dims[3]);
    Tensor <Tuple<Index, Scalar>, 4> output(dims[0], (dims[1] - extents[0]) / stride + 1, (dims[2] - extents[1]) / stride + 1, dims[3]);

    // here the mask contains (x, y) coordinates of the original argmax
    Tensor <Tuple<Index, Index>, 4> mask(output.dimensions());

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

            Index idx_flat = local_pool(k, j).first();
            Index idx_col = idx_flat / lengths.dimension(3) % lengths.dimension(2);
            Index idx_row = idx_flat / lengths.dimension(3) / lengths.dimension(2) % lengths.dimension(1);

            output(k, output_starts[1], output_starts[2], j) = local_pool(k, j).second();
            mask(k, output_starts[1], output_starts[2], j) = Tuple(idx_row, idx_col);
          }
        }
      }
    }
    return Tuple(output, mask);
  }

  template <typename Scalar, int Dim>
  inline Tensor<Scalar, Dim> do_max_pool_backward(const Tensor<Scalar, Dim>& grads, const Tensor<Scalar, Dim>& mask,
    const array<Index, Dim>& output_dims, const array<Index, Dim / 2>& extents, int stride) {
    return Tensor<Scalra, Dim>();
  }

  template <typename Scalar>
  inline Tensor<Scalar, 2> do_max_pool_backward(const Tensor<Scalar, 2>& grads, const Tensor<Scalar, 2>& mask,
    const array<Index, 2>& output_dims, const array<Index, 1>& extents, int stride) {

    array<Index, 2> starts({ 0, 0 });
    array<Index, 2> lengths({ output_dims[0], extents[0] });

    array<Index, 2> grad_starts({ 0, 0 });

    Tensor<Scalar, 2> output(output_dims);
    output.setZero();

    for (; starts[1] + extents[0] <= output_dims[1]; starts[1] += stride; grad_starts[1]++) {
      // unroll the index of the gradient being passed
      for (int k = 0; k < output_dims[0]; k++) {

        // index has been unrolled during the forward operation
        output.slice(starts, lengths)(k, mask(k, grad_starts[1])) = grads(k, grad_starts[1]);
      }
    }
    return output;
  }

  //backward 3d
  template <typename Scalar>
  inline Tensor<Scalar, 4> do_max_pool_backward(const Tensor<Scalar, 4>& grads, const Tensor<Scalar, 4>& mask,
    const array<Index, 4>& output_dims, const array<Index, 2>& extents, int stride) {

    array<Index, 4> starts({ 0, 0, 0, 0 });
    array<Index, 4> lengths({ output_dims[0], extents[0] });
    
    array<Index, 4> grad_starts({ 0, 0, 0, 0 });

    Tensor<Scalar, 4> output(output_dims);

    output.setZero();

    for (; starts[1] + extents[0] <= output_dims[1]; starts[1] += stride; grad_starts[1]++) {
      for (; starts[2] + extents[1] <= output_dims[2]; starts[2] += stride, grad_starts[2]++) {
        // unroll the index of the gradient being passed
        for (int k = 0; k < output_dims[0]; k++) {
          for (int j = 0; j < output_dims[3]; j++) {

            // index has been unrolled during the forward operation
            Index idx_row = mask(k, grad_starts[1]).first();
            Index idx_col = mask(k, grad_starts[1]).second();

            output.slice(starts, lengths)(k, idx_row, idx_col, j) = grads(k, grad_starts[1], grad_starts[2], j);
          }
        }
      }
    }
    return output;
  }
}