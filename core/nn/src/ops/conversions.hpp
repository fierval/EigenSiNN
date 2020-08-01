#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include <any>
#include <stdexcept>

namespace EigenSinn {
  template<typename T>
  using  MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  template<typename Scalar, int rank, typename sizeType>
  inline auto Tensor_to_Matrix(const Eigen::Tensor<Scalar, rank>& tensor, const sizeType rows, const sizeType cols) {
    return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols);
  }

  template<typename Scalar, typename sizeType>
  inline auto Tensor_to_Matrix(const Eigen::Tensor<Scalar, 1>& tensor, const sizeType cols) {
    return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), 1 , cols);
  }

  inline auto Tensor_to_Matrix(const Eigen::Tensor<float, 2>& tensor) {
    return Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
  }

  template<typename Scalar, typename... Dims>
  inline auto Matrix_to_Tensor(const MatrixType<Scalar>& matrix, Dims... dims) {
    constexpr int rank = sizeof... (Dims);
    return Eigen::TensorMap<Eigen::Tensor<const Scalar, rank>>(matrix.data(), { dims... });
  }

  template<typename Scalar, Index Rank>
  inline Tensor<Scalar, Rank> from_any(std::any t) {
    return std::any_cast<Tensor<Scalar, Rank>&>(t);
  }

  template<typename Scalar>
  inline Scalar from_any_scalar(std::any t) {
    return std::any_cast<Scalar&>(t);
  }

  template<typename Scalar>
  inline auto from_binary_to_category(const Tensor<Scalar, 2>& inp) {

    Tensor<Scalar, 1> cat(inp.dimension(0));

    for (Index row = 0; row < inp.dimension(0); row++) {
      for (Index col = 0; col < inp.dimension(1); col++) {
        if (inp(row, col) > 0) {
          cat(row) = col;
          break;
        }
        if (col == inp.dimension(1)) {
          throw std::logic_error("row without non-zero element");
        }
      }
    }

    return cat;
  }
}