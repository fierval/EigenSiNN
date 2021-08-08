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

    typedef std::vector<PtrTensorAdaptor<Scalar, Device_>> TensorAdaptorVector;
    typedef std::unordered_map<std::string, PtrTensorAdaptor<Scalar, Device_>> LayerTensorAdaptorMap;


    virtual void init() {};

    virtual void forward(PtrTensorAdaptor<Scalar, Device_>& prev_layer_base) = 0;

    // REVIEW: most layers have a single input, but not necessarily. Presence of the overloaded forward is legacy
    // many inputs going into one node
    virtual void forward(LayerTensorAdaptorMap& inputs) {
      if (inputs.size() == 1) {
        forward(inputs.begin()->second);
        return;
      }

      // otherwise override!
      assert(false);

    };

    virtual void backward(PtrTensorAdaptor<Scalar, Device_>& prev_layer, PtrTensorAdaptor<Scalar, Device_> next_layer_grad_any) = 0;

    // REVIEW: see above
    virtual void backward(LayerTensorAdaptorMap& prev_layer, PtrTensorAdaptor<Scalar, Device_>& next_layer_grad_any) {

      if (prev_layer.size() == 1) {
        backward(prev_layer.begin()->second, next_layer_grad_any);
        return;
      }

      // otherwise override!
      assert(false);
    }

    virtual void backward(LayerTensorAdaptorMap& prev_layer, TensorAdaptorVector& next_layer_grad) {

      if (next_layer_grad.size() == 1) {
        backward(prev_layer, next_layer_grad[0]);
        return;
      }

      // otherwise - add tensor adapters up!
      PtrTensorAdaptor<Scalar, Device_> acc(new TensorAdaptor<Scalar, Device_>(next_layer_grad[0]->get_dims()));
      acc->setZero();

      for (auto& ta : next_layer_grad) {
        *acc += *ta;
      }
    }

    virtual  PtrTensorAdaptor<Scalar, Device_> get_output() = 0;
    // optimization for a layer -> layer simple connection
    virtual  PtrTensorAdaptor<Scalar, Device_> get_loss_by_input_derivative() = 0;

    // generally we'll want the derivative by a given input
    virtual PtrTensorAdaptor<Scalar, Device_> get_loss_by_input_derivative(std::string& layer_name) {
      return empty_tensor();
    }

    virtual  PtrTensorAdaptor<Scalar, Device_> get_loss_by_weights_derivative() { return empty_tensor(); };
    virtual  PtrTensorAdaptor<Scalar, Device_> get_weights() { return empty_tensor(); };

    virtual  PtrTensorAdaptor<Scalar, Device_> get_loss_by_bias_derivative() { return empty_tensor(); }
    virtual  PtrTensorAdaptor<Scalar, Device_> get_bias() { return empty_tensor(); }

    virtual void set_weights(PtrTensorAdaptor<Scalar, Device_>&) {}
    virtual void set_bias(PtrTensorAdaptor<Scalar, Device_>&) {}

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
      PtrTensorAdaptor<Scalar, Device_> weights = get_weights();
      if (!weights) {
        return 0;
      }
      return weights->get_dims().size();
    }

    // graph traversal identifies LayerBase vs OpLayerBase based on this flag
    const bool is_multi_input() { return has_multiple_inputs; }

    inline static PtrTensorAdaptor<Scalar, Device_> empty_tensor() {
      return PtrTensorAdaptor<Scalar, Device_>();
    }

  protected:

    TensorAdaptorVector get_inputs_from_map(LayerTensorAdaptorMap& inputs) {
      assert(inputs.size() == 2);

      auto it = inputs.begin();
      PtrTensorAdaptor<Scalar, Device_> input1 = it->second;
      PtrTensorAdaptor<Scalar, Device_> input2 = (it + 1)->second;

      return std::vector{ input1, input2 };
    }

    std::vector<std::string> get_names_from_map(LayerTensorAdaptorMap& inputs) {
      assert(inputs.size() == 2);

      auto it = inputs.begin();
     std::string input1 = it->first;
     std::string input2 = (it + 1)->first;

     return std::vector{ input1, input2 };
    }

    bool is_cudnn = true, has_multiple_inputs=false;

    std::string op_name;
    std::string layer_name;
  };
}