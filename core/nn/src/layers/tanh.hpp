#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"

#ifdef __CUDACC__
#include "cudnn/cudnn_activations.hpp"
#endif

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class Tanh : public LayerBase<Scalar, Device_> {
  public:
    // leaky relu if necessary
    Tanh(bool _is_cudnn = false) 
      : inited(false)
      , is_cudnn(_is_cudnn) {
    
      assert(!is_cudnn || (Layout == RowMajor && Rank > 2));
    }

    void forward(LayerBase<Scalar, Device_>& prev_layer_any) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> prev_layer(prev_layer_any.get_output());

      // we have never initialized or switched from train to test
      // initialize the "1" tensor used for sigmoid backprop
      if (!inited || prev_layer.dimension(0) != layer_output.dimension(0)) {
        inited = true;
        init_cached(prev_layer);
      }

#ifdef __CUDACC__
      // no need to allocate output memory if we aren't using cudnn
      // as the op will allocate it for us
      if (is_cudnn && !cudnn_act) {
        layer_output.resize(prev_layer.dimensions());
        layer_grad.resize(prev_layer.dimensions());

        cudnn_act = std::make_shared<CudnnActivations<Scalar, Rank>>(prev_layer.dimensions(), cudnn_act_mode, 0.0);
      }

      if (is_cudnn) {
        cudnn_act->forward(prev_layer->data(), layer_output->data());
        return;
      }
#endif // EIGNE_USE_GPU


      layer_output.view() = prev_layer->tanh();
    }


    void backward(LayerBase<Scalar, Device_>& prev_layer_any, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> next_layer_grad(next_layer_grad_any);

#ifdef __CUDACC__
      if (is_cudnn) {
        cudnn_act->backward(next_layer_grad->data(), layer_grad->data());
        return;
      }
#endif

      layer_grad.view() = *next_layer_grad * (*ones - layer_output->pow(2.));
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
      return layer_output.raw();
    };

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() override {
      return layer_grad.raw();
    };

  private:
    void init_cached(const DeviceTensor<Scalar, Rank, Device_, Layout>& prev_layer)
    {
      ones.resize(prev_layer.dimensions());
      ones.setConstant(1);

      layer_output.resize(prev_layer.dimensions());
      layer_grad.resize(prev_layer.dimensions());
    }

    bool inited;
    bool is_cudnn;

#ifdef __CUDACC__
    cudnnActivationMode_t cudnn_act_mode = CUDNN_ACTIVATION_TANH;
    std::shared_ptr<CudnnActivations<Scalar, Rank>> cudnn_act;
#endif // __CUDACC__

    DeviceTensor<Scalar, Rank, Device_, Layout> ones, layer_output, layer_grad;
  };


}