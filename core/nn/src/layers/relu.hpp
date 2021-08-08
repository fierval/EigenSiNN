#pragma once

#include "layer_base.hpp"
#include "ops/relu.hpp"
#include "onnx/op_defs.h"

#ifdef __CUDACC__
#include "cudnn/cudnn_activations.hpp"
#endif

using namespace  Eigen;

namespace EigenSinn {

  template <typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class LeakyReLU : public LayerBase<Scalar, Device_> {
  public:
    // leaky relu if necessary
    LeakyReLU(float _thresh = 0.01)
      : LayerBase<Scalar, Device_>(OnnxOpNames::leakyrelu_op)
      , thresh(_thresh)  {
      is_cudnn = false;
    }

    void forward(PtrTensorAdaptor<Scalar, Device_>& prev_layer) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> x(prev_layer);

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

    void backward(PtrTensorAdaptor<Scalar, Device_>& prev_layer, PtrTensorAdaptor<Scalar, Device_> next_layer_grad) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> y(next_layer_grad);
#ifdef __CUDACC__
      if (is_cudnn) {
        cudnn_act->backward(y->data(), layer_grad->data());
        return;
      }
#endif
      layer_grad = leaky_relu_back(y, mask);
    }

    PtrTensorAdaptor<Scalar, Device_> get_output() override {
      return layer_output.raw();
    };

    PtrTensorAdaptor<Scalar, Device_> get_loss_by_input_derivative() override {
      return layer_grad.raw();
    };

    inline void set_cudnn(bool _is_cudnn) override {

      if (Rank < 4) { return; }
      // no LeakyReLU implementation in cuDNN
      assert(!_is_cudnn || (Layout == RowMajor && Rank > 2 && thresh == 0));
      is_cudnn = _is_cudnn;
    }

    // Save to ONNX
    const std::string add_onnx_node(EigenModel& model, const std::string& input_name) override {

      // https://github.com/onnx/onnx/blob/v1.9.0/docs/Operators.md#Relu
      auto * node = model.add_graph_node(get_layer_name(), get_op_name(), input_name);

      model.add_attr(node, "alpha", thresh);
      // save rank, not part of ONNX but necessary for loading
      model.add_attr(node, "rank", Rank);

      return node->output().Get(0);
    }

    const std::vector<Index> onnx_out_dims() override {
      return layer_output.vec_dims();
    }

  protected:

    Scalar thresh;
    DeviceTensor<Scalar, Rank, Device_, Layout> mask;
    DeviceTensor<Scalar, Rank, Device_, Layout> layer_output, layer_grad;

#ifdef __CUDACC__
    cudnnActivationMode_t cudnn_act_mode = CUDNN_ACTIVATION_RELU;
    std::shared_ptr<CudnnActivations<Scalar, Rank>> cudnn_act;
#endif // __CUDACC__
  };

  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class ReLU : public LeakyReLU<Scalar, Rank, Device_, Layout> {
  public:
    ReLU() : LeakyReLU<Scalar, Rank, Device_, Layout>(0) {
      op_name = std::string(OnnxOpNames::relu_op);
      is_cudnn = Rank > 2 && Layout == RowMajor;
    }

    const std::string add_onnx_node(EigenModel& model, const std::string& input_name) override {
      // https://github.com/onnx/onnx/blob/v1.9.0/docs/Operators.md#Relu
      auto * node = model.add_graph_node(get_layer_name(), get_op_name(), input_name);

      model.add_attr(node, "rank", Rank);

      return node->output().Get(0);
    }
  };

}