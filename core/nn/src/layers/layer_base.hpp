#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <ops/opsbase.hpp>
#include <ops/conversions.hpp>
#include <device/tensor_view.hpp>

using namespace Eigen;

namespace EigenSinn {

  enum class DebugInit : int {
    False = 0,
    Weights = 1,
    Bias = 2
  };

  template <typename Device_, typename Scalar, Index InRank, Index OutRank, int Layout=ColMajor>
  class LayerBase {

  public:
    virtual void init() {};

    virtual void forward(LayerBase<Scalar>& prev_layer_base) = 0;

    virtual void backward(LayerBase<Scalar>& prev_layer, LayerBase<Scalar>& next_layer_grad) { 
      return backward(prev_layer, next_layer_grad.get_loss_by_input_derivative()); }

    virtual void backward(LayerBase<Scalar>& prev_layer, Scalar * next_layer_grad) = 0;

    virtual  DeviceTensor<Device_, Scalar, OutRank, Layout>& get_output() {
      return layer_output;
    }

    virtual  DeviceTensor<Device_, Scalar, InRank, Layout>& get_loss_by_input_derivative() {
      return layer_gradient;
    }

    virtual  DeviceTensor<Device_, Scalar, InRank, Layout>& get_loss_by_weights_derivative() { return DeviceTensor(); };
    virtual  DeviceTensor<Device_, Scalar, InRank, Layout>& get_weights() { return DeviceTensor(); };

    virtual  DeviceTensor<Device_, Scalar, 1, Layout>& get_loss_by_bias_derivative() { return DeviceTensor(); }
    virtual  DeviceTensor<Device_, Scalar, 1, Layout>& get_bias() { return DeviceTensor(); }

    DSizes<Index, InRank>& get_in_dims() { return in_dims; }
    DSizes<Index, OutRank>& get_out_dims() { return out_dims; }

    DSizes<Index, 1>& get_bias_dims() { return bias_dims; }
    DSizes<Index, InRank>& get_weight_dims() { return weight_dims; }

    void set_in_dims(DSizes<Index, InRank>& _in_dims) { in_dims = _in_dims; }
    void set_out_dims(DSizes<Index, OutRank>& _out_dims) { out_dims = _out_dims; }

    void set_dims(DSizes<Index, InRank>& _in_dims, DSizes<Index, OutRank>& _out_dims) {
      set_in_dims(_in_dims);
      set_out_dims(_out_dims);
    }

    void set_dims(LayerBase<Scalar>& layer) {
      if (are_dims_unset(layer.get_out_dims())) {
        set_dims(layer.get_out_dims(), layer.get_out_dims());
      }
    }

    void set_bias_dims(DSizes<Index, 1>& _bias_dims) { bias_dims = _bias_dims; }
    void set_weight_dims(DSizes<Index, InRank>& _weight_dims) { weight_dims = _weight_dims; }

    virtual ~LayerBase() = default;

  protected:
    DSizes<Index, InRank> in_dims, weight_dims;
    DSizes<Index, OutRank> out_dims; 
    DSizes<Index, 1> bias_dims;

    DeviceTensor<Device_, Scalar, OutRank, Layout> layer_output;
    DeviceTensor<Device_, Scalar, InRank, Layout> layer_gradient;
    DeviceTensor<Device_, Scalar, InRank, Layout> weights;
    DeviceTensor<Device_, Scalar, 1, Layout> bias;

    bool are_dims_unset(std::vector<Index>& dims) { return in_dims.size() == 0 || dims[0] != in_dims[0]; }

  };
}