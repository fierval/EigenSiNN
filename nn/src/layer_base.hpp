#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

class LayerBase {

public:
  
  virtual void forward(const MatrixXd& prev_layer) = 0;

  virtual void backward(const MatrixXd& prev_layer_out, const MatrixXd& next_layer_grad) = 0;
};