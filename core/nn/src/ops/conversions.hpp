#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

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

}