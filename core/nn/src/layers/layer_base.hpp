#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <ops/opsbase.hpp>
#include <ops/conversions.hpp>
#include <device/device_tensor.hpp>
#include <onnx/model.h>

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
    LayerBase(const char* _op_name) : op_name(_op_name) {

    }

    virtual void init() {};

    virtual void forward(PtrTensorAdapter<Scalar, Device_>& prev_layer_base) = 0;

    // REVIEW: most layers have a single input, but not necessarily. Presence of the overloaded forward is legacy
    virtual void forward(std::vector<PtrTensorAdapter<Scalar, Device_>>& inputs) {
      if (inputs.size() == 1) {
        forward(inputs[0]);
        return;
      }

      // otherwise override!
      assert(false);
    }
    virtual void backward(PtrTensorAdapter<Scalar, Device_>& prev_layer, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) = 0;

    // REVIEW: see above
    virtual void backward(std::vector<PtrTensorAdapter<Scalar, Device_>>& prev_layer, PtrTensorAdapter<Scalar, Device_>& next_layer_grad_any) {

      if (prev_layer.size() == 1) {
        backward(prev_layer[0], next_layer_grad_any);
        return;
      }

      // otherwise override!
      assert(false);
    }

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
    virtual const std::string add_onnx_node(EigenModel& model, std::vector<std::string>& input_names) {

      if (input_names.size() == 1) {
        return add_onnx_node(model, input_names[0]);
        
      }

      // override!
      throw std::logic_error("Not implemented");
    }

    virtual const std::vector<Index> onnx_out_dims() { return std::vector<Index>(); }

    virtual void load_onnx_data(EigenModel& model, std::vector<std::string>& inputs) {}

    std::string& get_op_name() { return op_name; }
    std::string& get_layer_name() { return layer_name; }

    void set_layer_name(const std::string& name) {
      assert(!name.empty());
      layer_name = name;
    }

    void set_layer_name(const std::string&& name) {
      assert(!name.empty());
      layer_name = name;
    }

    // dimensions of the weights for the optimizer
    virtual Index get_optimizer_rank() {
      PtrTensorAdapter<Scalar, Device_> weights = get_weights();
      if (!weights) {
        return 0;
      }
      return weights->get_dims().size();
    }

  protected:
    bool is_cudnn = false;
    std::string op_name;
    std::string layer_name;
  };
}