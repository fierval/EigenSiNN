#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

#ifdef __CUDACC__
#include "cudnn/cudnn_activations.hpp"
#endif

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class LeakyReLU : public LayerBase<Scalar, Device_> {
  public:
    // leaky relu if necessary
    LeakyReLU(float _thresh = 0.01, bool _is_cudnn = false)
      : thresh(_thresh) {

      set_cudnn(_is_cudnn);
    }

    void forward(LayerBase<Scalar, Device_>& prev_layer) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> x(prev_layer.get_output());

#ifdef __CUDACC__
      // no need to allocate output memory if we aren't using cudnn
      // as the op will allocate it for us
      if (is_cudnn && !cudnn_act) {
        layer_output.resize(x.dimensions());
        layer_grad.resize(x.dimensions());

        cudnn_act = std::make_shared<CudnnActivations<Scalar, Rank>>(x.dimensions(), cudnn_act_mode, thresh);
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
#ifdef __CUDACC__
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

    inline void set_cudnn(bool _is_cudnn) override {
      
      // no LeakyReLU implementation in cuDNN
      assert(!_is_cudnn || (Layout == RowMajor && Rank > 2 && thresh == 0));
      is_cudnn = _is_cudnn;
    }

  protected:
    float thresh;
    DeviceTensor<Scalar, Rank, Device_, Layout> mask;
    DeviceTensor<Scalar, Rank, Device_, Layout> layer_output, layer_grad;
    bool is_cudnn;

#ifdef __CUDACC__
    cudnnActivationMode_t cudnn_act_mode = CUDNN_ACTIVATION_RELU;
    std::shared_ptr<CudnnActivations<Scalar, Rank>> cudnn_act;
#endif // __CUDACC__

  };

  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class ReLU : public LeakyReLU<Scalar, Rank, Device_, Layout> {
  public:
    ReLU(bool _is_cudnn = false) : LeakyReLU<Scalar, Rank, Device_, Layout>(0, _is_cudnn) {


    }
  };

}