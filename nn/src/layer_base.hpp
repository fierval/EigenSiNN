#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

class LayerBase {

public:
  
  void adjust_linear_bias(MatrixXd& layer) {
      layer.conservativeResize(layer.rows(), layer.cols() + 1);
      layer.col(layer.cols() - 1) = VectorXd::Ones(layer.rows());
  }

  virtual void forward(MatrixXd& prev_layer) = 0;

  virtual void backward(const MatrixXd& prev_layer, const MatrixXd& next_layer_grad) = 0;
};