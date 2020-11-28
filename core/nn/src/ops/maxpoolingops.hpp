#pragma once

#include "opsbase.hpp"
#include <device/device_tensor.hpp>

namespace EigenSinn {

  template <int Rank, class Dims>
  inline bool check_valid_params(const array<Index, Rank / 2>& extents, int stride, Dims& dims) {

    if (stride <= 0) {
      return false;
    }

    if (Rank != 2 && Rank != 4) {
      return false;
    }

    for (Index i = 0; i < Rank / 2; i++) {
      // we are interested in the second or 3rd and 4th
      // depending on the tensor: 2d or 4d
      int tensor_dim = Rank == 2 ? dims[i + 1] : dims[i + 2];
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

  template <typename Scalar, int Rank, int Layer, typename Device_>
  struct MaxPooler {};


  template <typename Scalar, typename Device_>
  struct MaxPooler<Scalar, 2, ColMajor, Device_> {

    inline auto do_max_pool(const DeviceTensor<Device_, Scalar, 2, ColMajor>& t, const array<Index, 1>& extents, int stride) {
      auto dims = t.dimensions();

      if (!check_valid_params<2>(extents, stride, dims)) {

        throw std::invalid_argument("Invalid pooling dimensions");
      }

      // we get the index as well as the value so we can create a mask
      DeviceTensor<Device_, Tuple<Index, Scalar>, 1> local_pool(dims[0]);
      DeviceTensor <Device_, Scalar, 2> output(dims[0], (dims[1] - extents[0]) / stride + 1);
      DeviceTensor <Device_, Index, 2> mask(output.dimensions());

      array<Index, 2> starts({ 0, 0 });
      array<Index, 2> lengths({ dims[0], extents[0] });
      array<Index, 1> reduce_dims({ 1 });

      array<Index, 2> output_starts({ 0, 0 });

      DeviceTensor <Device_, Tuple<Index, Scalar>, 2> index_tuples(lengths);

      for (starts[1] = 0, output_starts[1] = 0; starts[1] + extents[0] <= dims[1]; starts[1] += stride, output_starts[1]++) {

        index_tuples.view() = t->slice(starts, lengths).index_tuples();

        // get the maximums and their indices
        local_pool.view() = index_tuples->reduce(reduce_dims, internal::ArgMaxTupleReducer<Tuple<Index, Scalar>>());
        // split the tuple into its arrays: value and index of the input where
        // gradient will be propagated relative to the current slice
        for (int k = 0; k < local_pool.dimension(0); k++) {
          
          output->operator()(k, output_starts[1]) = (*local_pool)(k).second;
          mask->operator()(k, output_starts[1]) = (*local_pool)(k).first;
        }
      }

      return Tuple(output, mask);

    }

    inline DeviceTensor<Device_, Scalar, 2, ColMajor> do_max_pool_backward(const DeviceTensor<Device_, Scalar, 2>& grads, const DeviceTensor<Device_, Index, 2>& mask,
      const array<Index, 2>& original_dims, const array<Index, 1>& extents, int stride) {

      array<Index, 2> starts({ 0, 0 });
      array<Index, 2> lengths({ original_dims[0], extents[0] });

      array<Index, 2> grad_starts({ 0, 0 });

      DeviceTensor<Device_, Scalar, 2> output(original_dims);
      output.setZero();

      for (starts[1] = 0, grad_starts[1] = 0; starts[1] + extents[0] <= original_dims[1]; starts[1] += stride, grad_starts[1]++) {
        // unroll the index of the gradient being passed
        for (int k = 0; k < original_dims[0]; k++) {

          Index idx_flat = (*mask)(k, grad_starts[1]);
          Index idx_col = (idx_flat - k) / lengths[0] % lengths[1];

          // index has been unrolled during the forward operation
          (*output)(starts[0] + k, starts[1] + idx_col) += (*grads)(k, grad_starts[1]);
        }
      }
      return output;
    }
  };

  template <typename Scalar, typename Device_>
  struct MaxPooler<Scalar, 4, ColMajor, Device_> {
    inline auto do_max_pool(const DeviceTensor<Device_, Scalar, 4>& t, const array<Index, 2>& extents, int stride) {
      auto dims = t.dimensions();

      if (!check_valid_params<4>(extents, stride, dims)) {

        throw std::invalid_argument("Invalid pooling dimensions");
      }

      DeviceTensor<Device_, Tuple<Index, Scalar>, 2> local_pool(dims[0], dims[1]);
      
      DeviceTensor<Device_, Scalar, 4> output(dims[0], dims[1], (dims[2] - extents[0]) / stride + 1, (dims[3] - extents[1]) / stride + 1);

      DeviceTensor<Device_, Index, 4> mask(output.dimensions());

      array<Index, 4> starts({ 0, 0, 0, 0 });
      array<Index, 4> lengths({ dims[0], dims[1], extents[0], extents[1]});
      array<Index, 2> reduce_dims({ 2, 3 });

      array<Index, 4> output_starts({ 0, 0, 0, 0 });

      // get index tuples in order to use tuple reducer
      DeviceTensor<Device_, Tuple<Index, Scalar>, 4> index_tuples(lengths);

      for (starts[2] = 0, output_starts[2] = 0; starts[2] + extents[0] <= dims[2]; starts[2] += stride, output_starts[2]++) {
        for (starts[3] = 0, output_starts[3] = 0; starts[3] + extents[1] <= dims[3]; starts[3] += stride, output_starts[3]++) {

          index_tuples.view() = t->slice(starts, lengths).index_tuples();

          // get pooling results
          local_pool.view() = index_tuples->reduce(reduce_dims, internal::ArgMaxTupleReducer<Tuple<Index, Scalar>>());

          // unchain indices. TODO: unwinding them in the backward pass will be harder than 2 dims
          for (int k = 0; k < local_pool.dimension(0); k++) {
            for (int j = 0; j < local_pool.dimension(1); j++) {

              (*mask)(k, j, output_starts[2], output_starts[3]) = (*local_pool)(k, j).first;
              (*output)(k, j, output_starts[2], output_starts[3]) = (*local_pool)(k, j).second;
            }
          }
        }
      }
      return Tuple(output, mask);
    }

    inline DeviceTensor<Device_, Scalar, 4> do_max_pool_backward(const DeviceTensor<Device_, Scalar, 4>& grads, const Tensor<Index, 4>& mask,
      const array<Index, 4>& original_dims, const array<Index, 2>& extents, int stride) {

      array<Index, 4> starts({ 0, 0, 0, 0 });
      array<Index, 4> lengths({ original_dims[0], original_dims[1], extents[0], extents[1]});

      array<Index, 4> grad_starts({ 0, 0, 0, 0 });

      DeviceTensor<Device_, Scalar, 4> output(original_dims);

      output.setZero();

      for (starts[2] = 0, grad_starts[2] = 0; starts[2] + extents[0] <= original_dims[2]; starts[2] += stride, grad_starts[2]++) {
        for (starts[3] = 0, grad_starts[3] = 0; starts[3] + extents[1] <= original_dims[3]; starts[3] += stride, grad_starts[3]++) {
          // unroll the index of the gradient being passed
          for (int k = 0; k < original_dims[0]; k++) {
            for (int j = 0; j < original_dims[1]; j++) {

              // index has been unrolled during the forward operation
              Index idx_flat = (*mask)(k, j, grad_starts[2], grad_starts[3]);

              // extract column-major order based on: https://en.wikipedia.org/wiki/Row-_and_column-major_order
              Index idx_plane = ((idx_flat - k) / lengths[0] - j) / lengths[1];
              Index idx_row = idx_plane % lengths[2];
              Index idx_col = (idx_plane - idx_row) / lengths[2];

              (*output)(k, j, starts[2] + idx_row, starts[3] + idx_col) += (*grads)(k, j, grad_starts[2], grad_starts[3]);
            }
          }
        }
      }
      return output;
    }


  };
}