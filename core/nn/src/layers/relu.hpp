#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

#ifdef EIGEN_USE_GPU
#include "cudnn/cudnn_activations.hpp"
#endif

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class LeakyReLU : public LayerBase<Scalar, Device_> {
  public:
    // leaky relu if necessary
    LeakyReLU(float _thresh = 0.01, bool _is_cudnn = false)
      : thresh(_thresh)
      , is_cudnn(_is_cudnn) {

    }

    void forward(LayerBase<Scalar, Device_>& prev_layer) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> x(prev_layer.get_output());
      auto res = leaky_relu(x, thresh);

      layer_output = res.second;
      mask = res.first;
    }

    void backward(LayerBase<Scalar, Device_>& prev_layer, PtrTensorAdapter<Scalar, Device_> next_layer_grad) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> x(next_layer_grad);

      layer_grad = leaky_relu_back(x, mask);
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
      return layer_output.raw();
    };

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() override {
      return layer_grad.raw();
    };


  private:
    float thresh;
    DeviceTensor<Scalar, Rank, Device_, Layout> mask;
    DeviceTensor<Scalar, Rank, Device_, Layout> layer_output, layer_grad;
    bool is_cudnn;

#ifdef EIGEN_USE_GPU
    cudnnTensorDescriptor_t tensor_desc;
#endif // EIGEN_USE_GPU

  };

  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class ReLU : public LeakyReLU<Scalar, Rank, Device_, Layout> {
  public:
    ReLU() : LeakyReLU<Scalar, Rank, Device_, Layout>(0) {}
  };

}