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
  inline Tensor<Scalar, Rank> do_pool(Tensor<Scalar, Rank>& t, const array<int, Rank / 2>& extents, int stride) {

    return Tensor<Scalar, Rank>();
  }

  template <typename Scalar>
  inline Tensor<Scalar, 2> do_pool(Tensor<Scalar, 2>& t, const array<int, 1>& extents, int stride) {
    auto dims = t.dimensions();

    if (!check_valid_params<2>(extents, stride, dims)) {

      throw std::invalid_argument("Invalid pooling dimensions");
    }

    // we get the index as well as the value so we can create a mask
    Tensor <Tuple<DenseIndex, Scalar>, 1> arg_max_tuple(dims[0]);
    Tensor <Scalar, 2> output(dims[0], (dims[1] - extents[0]) / stride + 1);
    Tensor <byte, 2> mask(dims[0], dims[1])

    array<Index, 2> starts({ 0, 0 });
    array<Index, 2> lengths({ dims[0], extents[0] });
    array<Index, 1> reduce_dims({ 1 });

    array<Index, 2> output_starts({ 0, 0 });
    array<Index, 2> output_lengths({ output.dimension(0), 1 });

    for (int i = 0; i + extents[0] <= dims[1]; i++) {

      output.slice(output_starts, output_lengths) = 
        t.slice(starts, lengths).reduce(reduce_dims, internal::ArgMaxTupleReducer<Tuple<DenseIndex, Scalar>>());
      output_starts[1]++;
      starts[1] += extents[0];
    }

    return output;

  }

  template <typename Scalar>
  inline Tensor<Scalar, 4> do_pool(Tensor<Scalar, 4>& t, const array<int, 2>& extents, int stride) {
    auto dims = t.dimensions();

    if (!check_valid_params<2>(extents, stride, dims)) {

      throw std::invalid_argument("Invalid pooling dimensions");
    }

    Tensor <Tuple<DenseIndex, Scalar>, 4> arg_max_tuple(dims[0], dims[3]);
    Tensor <Tuple<DenseIndex, Scalar>, 4> output(dims[0], (dims[1] - extents[0]) / stride + 1, (dims[2] - extents[1]) / stride + 1);
    Tensor<Scalar, 4> mask(dims[0], dims[1], dims[2], dims[3]);

    array<Index, 4> starts({ 0, 0, 0, 0 });
    array<Index, 4> lengths({ dims[0], extents[0], extents[1], dims[3] });
    array<Index, 2> reduce_dims({ 1, 2 });

    array<Index, 4> output_starts({ 0, 0, 0, 0 });
    array<Index, 4> output_lengths({ output.dimension(0), 1, 1, output.dimension(3) });

    for (int i = 0; i + extents[0] <= dims[1]; i++) {
      for (int j = 0; j + extents[1] <= dims[2]; j++) {

        output.slice(output_starts, output_lengths) = t.slice(starts, lengths).reduce(reduce_dims, internal::ArgMaxTupleReducer <Tuple<DenseIndex, Scalar>());

        // move by column first
        output_starts[2]++; 
        starts[2] += extents[1];
      }
      output_starts[1]++;
      starts[1] += extents[0];
    }

    return output;

  }

}