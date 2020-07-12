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
  inline bool check_valid_params(const array<Index, Rank / 2>& extents, int stride, Dims& dims) {

    if (stride <= 0) {
      return false;
    }

    if ((Rank == 2 || Rank == 4) && Rank / 2 != extents.size()) {

      return false;
    }

    for (Index i = 0; i < Rank / 2; i++) {
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

  template <typename T, int Rank>
  struct MaxPooler {};


  template <typename Scalar>
  struct MaxPooler<Scalar, 2> {
    inline auto do_max_pool(Tensor<Scalar, 2>& t, const array<Index, 1>& extents, int stride) {
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

      Tensor < Tuple<Index, Scalar>, 2> index_tuples(lengths);

      for (starts[1] = 0, output_starts[1] = 0; starts[1] + extents[0] <= dims[1]; starts[1] += stride, output_starts[1]++) {

        index_tuples = t.slice(starts, lengths).index_tuples();

        // get the maximums and their indices
        local_pool = index_tuples.reduce(reduce_dims, internal::ArgMaxTupleReducer<Tuple<Index, Scalar>>());
        // split the tuple into its arrays: value and index of the input where
        // gradient will be propagated relative to the current slice
        for (int k = 0; k < local_pool.dimension(0); k++) {
          
          output(k, output_starts[1]) = local_pool(k).second;
          mask(k, output_starts[1]) = local_pool(k).first;
        }
      }

      return Tuple(output, mask);

    }

    inline Tensor<Scalar, 2> do_max_pool_backward(const Tensor<Scalar, 2>& grads, const Tensor<Index, 2>& mask,
      const array<Index, 2>& original_dims, const array<Index, 1>& extents, int stride) {

      array<Index, 2> starts({ 0, 0 });
      array<Index, 2> lengths({ original_dims[0], extents[0] });

      array<Index, 2> grad_starts({ 0, 0 });

      Tensor<Scalar, 2> output(original_dims);
      output.setZero();

      for (starts[1] = 0, grad_starts[1] = 0; starts[1] + extents[0] <= original_dims[1]; starts[1] += stride, grad_starts[1]++) {
        // unroll the index of the gradient being passed
        for (int k = 0; k < original_dims[0]; k++) {

          Index idx_flat = mask(k, grad_starts[1]);
          Index idx_col = (idx_flat - k) / lengths[0] % lengths[1];

          // index has been unrolled during the forward operation
          output(starts[0] + k, starts[1] + idx_col) += grads(k, grad_starts[1]);
        }
      }
      return output;
    }


  };

  template <typename Scalar>
  struct MaxPooler<Scalar, 4> {
    inline auto do_max_pool(Tensor<Scalar, 4>& t, const array<Index, 2>& extents, int stride) {
      auto dims = t.dimensions();

      if (!check_valid_params<4>(extents, stride, dims)) {

        throw std::invalid_argument("Invalid pooling dimensions");
      }

      Tensor<Tuple<Index, Scalar>, 2> local_pool(dims[0], dims[3]);
      
      Tensor<Scalar, 4> output(dims[0], (dims[1] - extents[0]) / stride + 1, (dims[2] - extents[1]) / stride + 1, dims[3]);

      Tensor<Index, 4> mask(output.dimensions());

      array<Index, 4> starts({ 0, 0, 0, 0 });
      array<Index, 4> lengths({ dims[0], extents[0], extents[1], dims[3] });
      array<Index, 2> reduce_dims({ 1, 2 });

      array<Index, 4> output_starts({ 0, 0, 0, 0 });

      // get index tuples in order to use tuple reducer
      Tensor<Tuple<Index, Scalar>, 4> index_tuples(lengths);

      for (starts[1] = 0, output_starts[1] = 0; starts[1] + extents[0] <= dims[1]; starts[1] += stride, output_starts[1]++) {
        for (starts[2] = 0, output_starts[2] = 0; starts[2] + extents[1] <= dims[2]; starts[2] += stride, output_starts[2]++) {

          index_tuples = t.slice(starts, lengths).index_tuples();

          // get pooling results
          local_pool = index_tuples.reduce(reduce_dims, internal::ArgMaxTupleReducer<Tuple<Index, Scalar>>());

          // unchain indices. TODO: unwinding them in the backward pass will be harder than 2 dims
          for (int k = 0; k < local_pool.dimension(0); k++) {
            for (int j = 0; j < local_pool.dimension(1); j++) {

              mask(k, output_starts[1], output_starts[2], j) = local_pool(k, j).first;
              output(k, output_starts[1], output_starts[2], j) = local_pool(k, j).second;
            }
          }
        }
      }
      return Tuple(output, mask);
    }

    inline Tensor<Scalar, 4> do_max_pool_backward(const Tensor<Scalar, 4>& grads, const Tensor<Index, 4>& mask,
      const array<Index, 4>& original_dims, const array<Index, 2>& extents, int stride) {

      array<Index, 4> starts({ 0, 0, 0, 0 });
      array<Index, 4> lengths({ original_dims[0], extents[0], extents[1], original_dims[3] });

      array<Index, 4> grad_starts({ 0, 0, 0, 0 });

      Tensor<Scalar, 4> output(original_dims);

      output.setZero();

      for (starts[1] = 0, grad_starts[1] = 0; starts[1] + extents[0] <= original_dims[1]; starts[1] += stride, grad_starts[1]++) {
        for (starts[2] = 0, grad_starts[2] = 0; starts[2] + extents[1] <= original_dims[2]; starts[2] += stride, grad_starts[2]++) {
          // unroll the index of the gradient being passed
          for (int k = 0; k < original_dims[0]; k++) {
            for (int j = 0; j < original_dims[3]; j++) {

              // index has been unrolled during the forward operation
              Index idx_flat = mask(k, grad_starts[1], grad_starts[2], j);

              // extract column-major order based on: https://en.wikipedia.org/wiki/Row-_and_column-major_order
              Index idx_row = (idx_flat - k) / lengths[0] % lengths[1];
              Index idx_col = ((idx_flat  - k) / lengths[0] - idx_row) / lengths[1] % lengths[2];

              output(k, starts[1] + idx_row, starts[2] + idx_col, j) += grads(k, grad_starts[1], grad_starts[2], j);
            }
          }
        }
      }
      return output;
    }


  };
}