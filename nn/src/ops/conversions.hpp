#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

namespace EigenSinn {
  template<typename T>
  using  MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  template<typename Scalar, int rank, typename sizeType>
  auto Tensor_to_Matrix(const Eigen::Tensor<Scalar, rank>& tensor, const sizeType rows, const sizeType cols) {
    return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols);
  }


  template<typename Scalar, typename... Dims>
  auto Matrix_to_Tensor(const MatrixType<Scalar>& matrix, Dims... dims) {
    constexpr int rank = sizeof... (Dims);
    return Eigen::TensorMap<Eigen::Tensor<const Scalar, rank>>(matrix.data(), { dims... });
  }

}