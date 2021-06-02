#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <ops/opsbase.hpp>
#include <ops/conversions.hpp>
#include <device/device_tensor.hpp>
#include <onnx/common.h>

using namespace Eigen;

namespace EigenSinn {

  enum class DebugInit : int {
    False = 0,
    Weights = 1,
    Bias = 2
  };

  template <typename Scalar, typename Device_ = ThreadPoolDevice>
  class LayerBase {

  public:
    virtual void init() {};

    virtual void forward(LayerBase<Scalar, Device_>& prev_layer_base) = 0;

    virtual void backward(LayerBase<Scalar, Device_>& prev_layer, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) = 0;

    virtual  PtrTensorAdapter<Scalar, Device_> get_output() = 0;

    virtual  PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() = 0;

    virtual  PtrTensorAdapter<Scalar, Device_> get_loss_by_weights_derivative() { return PtrTensorAdapter<Scalar, Device_>(); };
    virtual  PtrTensorAdapter<Scalar, Device_> get_weights() { return PtrTensorAdapter<Scalar, Device_>(); };

    virtual  PtrTensorAdapter<Scalar, Device_> get_loss_by_bias_derivative() { return PtrTensorAdapter<Scalar, Device_>(); }
    virtual  PtrTensorAdapter<Scalar, Device_> get_bias() { return PtrTensorAdapter<Scalar, Device_>(); }

    virtual void set_weights(PtrTensorAdapter<Scalar, Device_>&) {}
    virtual void set_bias(PtrTensorAdapter<Scalar, Device_>&) {}

    virtual ~LayerBase() = default;
    
    virtual void set_cudnn(bool _is_cudnn) {}

    virtual const std::string add_onnx_node(EigenModel& model, const std::string& input_name) { return std::string(""); }
    virtual const std::vector<Index> onnx_out_dims() { return std::vector<Index>(); }

    virtual void load_onnx_data(EigenModel& model, std::vector<std::string>& inputs) {}
  };
}