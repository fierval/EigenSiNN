#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <ops/opsbase.hpp>
#include <ops/conversions.hpp>
#include <device/device_tensor.hpp>
#include <onnx/model.h>

using namespace Eigen;

namespace EigenSinn {

  template <typename Scalar, typename Device_ = ThreadPoolDevice>
  class LayerBase {

  public:
    LayerBase(const char* _op_name) : op_name(_op_name), layer_name("") {

    }

    typedef std::vector<PtrTensorAdapter<Scalar, Device_>> TensorAdapterVector;
    typedef std::unordered_map<std::string, PtrTensorAdapter<Scalar, Device_>> LayerTensorAdapterMap;


    virtual void init() {};

    virtual void forward(PtrTensorAdapter<Scalar, Device_>& prev_layer_base) = 0;

    // REVIEW: most layers have a single input, but not necessarily. Presence of the overloaded forward is legacy
    // many inputs going into one node
    virtual void forward(LayerTensorAdapterMap& inputs) {
      if (inputs.size() == 1) {
        forward(inputs.begin()->second);
        return;
      }

      // otherwise override!
      assert(false);

    };

    virtual void backward(PtrTensorAdapter<Scalar, Device_>& prev_layer, PtrTensorAdapter<Scalar, Device_> next_layer_grad_any) = 0;

    // REVIEW: see above
    virtual void backward(LayerTensorAdapterMap& prev_layer, PtrTensorAdapter<Scalar, Device_>& next_layer_grad_any) {

      if (prev_layer.size() == 1) {
        backward(prev_layer.begin()->second, next_layer_grad_any);
        return;
      }

      // otherwise override!
      assert(false);
    }

    virtual void backward(LayerTensorAdapterMap& prev_layer, TensorAdapterVector& next_layer_grad) {

      if (next_layer_grad.size() == 1) {
        backward(prev_layer, next_layer_grad[0]);
        return;
      }

      // otherwise - add tensor adapters up!
      PtrTensorAdapter<Scalar, Device_> acc(new TensorAdapter<Scalar, Device_>(next_layer_grad[0]->get_dims()));
      acc->setZero();

      for (auto& ta : next_layer_grad) {
        *acc += *ta;
      }
    }

    virtual  PtrTensorAdapter<Scalar, Device_> get_output() = 0;
    // optimization for a layer -> layer simple connection
    virtual  PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() = 0;

    // generally we'll want the derivative by a given input
    virtual PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivatives(std::string& layer_name) {
      return empty_tensor();
    }

    virtual std::vector<PtrTensorAdapter<Scalar, Device_>> get_loss_by_input_derivatives() {

      throw std::logic_error("Not implemented for this layer!");
    }

    virtual  PtrTensorAdapter<Scalar, Device_> get_loss_by_weights_derivative() { return empty_tensor(); };
    virtual  PtrTensorAdapter<Scalar, Device_> get_weights() { return empty_tensor(); };

    virtual  PtrTensorAdapter<Scalar, Device_> get_loss_by_bias_derivative() { return empty_tensor(); }
    virtual  PtrTensorAdapter<Scalar, Device_> get_bias() { return empty_tensor(); }

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

    // graph traversal identifies LayerBase vs OpLayerBase based on this flag
    const bool is_multi_input() { return has_multiple_inputs; }

    inline static PtrTensorAdapter<Scalar, Device_> empty_tensor() {
      return PtrTensorAdapter<Scalar, Device_>();
    }

  protected:
    bool is_cudnn = true, has_multiple_inputs=false;

    std::string op_name;
    std::string layer_name;
  };
}