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

      // no LeakyReLU implementation in cuDNN
      assert(!is_cudnn || (Layout == RowMajor && Rank > 2 && thresh == 0));

    }

    void forward(LayerBase<Scalar, Device_>& prev_layer) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> x(prev_layer.get_output());

#ifdef EIGEN_USE_GPU
      // no need to allocate output memory if we aren't using cudnn
      // as the op will allocate it for us
      if (is_cudnn && !tensor_desc) {
        layer_output.resize(x.dimensions());
        layer_grad.resize(x.dimensions());

        tensor_desc = std::make_shared<TensorDescWrapper<Rank>>(x.dimensions());

        cudnn_act = std::make_shared<CudnnActivations<Scalar>>(*tensor_desc, cudnn_act_mode, thresh);
      }

      if (is_cudnn) {
        cudnn_act->forward(x->data(), layer_output->data());
        return;
      }
#endif // EIGNE_USE_GPU

      auto res = leaky_relu(x, thresh);

      layer_output = res.second;
      mask = res.first;
    }

    void backward(LayerBase<Scalar, Device_>& prev_layer, PtrTensorAdapter<Scalar, Device_> next_layer_grad) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> y(next_layer_grad);
#ifdef EIGEN_USE_GPU
      if (is_cudnn) {
        cudnn_act->backward(y->data(), layer_grad->data());
        return;
      }
#endif
      layer_grad = leaky_relu_back(y, mask);
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
      return layer_output.raw();
    };

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() override {
      return layer_grad.raw();
    };



  protected:
    float thresh;
    DeviceTensor<Scalar, Rank, Device_, Layout> mask;
    DeviceTensor<Scalar, Rank, Device_, Layout> layer_output, layer_grad;
    bool is_cudnn;

#ifdef EIGEN_USE_GPU
    cudnnActivationMode_t cudnn_act_mode = CUDNN_ACTIVATION_RELU;
    std::shared_ptr<TensorDescWrapper<Rank>> tensor_desc;
    std::shared_ptr<CudnnActivations<Scalar>> cudnn_act;
#endif // EIGEN_USE_GPU

  };

  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class ReLU : public LeakyReLU<Scalar, Rank, Device_, Layout> {
  public:
    ReLU(bool _is_cudnn = false) : LeakyReLU<Scalar, Rank, Device_, Layout>(0, _is_cudnn) {


    }
  };

}