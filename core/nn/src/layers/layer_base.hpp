#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <ops/opsbase.hpp>
#include <ops/conversions.hpp>
#include <device/device_tensor.hpp>

using namespace Eigen;

namespace EigenSinn {

  enum class DebugInit : int {
    False = 0,
    Weights = 1,
    Bias = 2
  };

  template <typename Scalar>
  class LayerBase {

  public:
    virtual void init() {};

    virtual void forward(LayerBase<Scalar>& prev_layer_base) = 0;

    virtual void backward(LayerBase<Scalar>& prev_layer, std::any next_layer_grad_any) = 0;

    virtual  std::any get_output() = 0;

    virtual  std::any get_loss_by_input_derivative() = 0;

    virtual  std::any get_loss_by_weights_derivative() { return std::any(); };
    virtual  std::any get_weights() { return std::any(); };

    virtual  std::any get_loss_by_bias_derivative() { return std::any(); }
    virtual  std::any get_bias() { return std::any(); }

    //DSizes<Index, InRank>& get_in_dims() { return in_dims; }
    //DSizes<Index, OutRank>& get_out_dims() { return out_dims; }

    //DSizes<Index, 1>& get_bias_dims() { return bias_dims; }
    //DSizes<Index, InRank>& get_weight_dims() { return weight_dims; }

    //void set_in_dims(DSizes<Index, InRank>& _in_dims) { in_dims = _in_dims; }
    //void set_out_dims(DSizes<Index, OutRank>& _out_dims) { out_dims = _out_dims; }

    //void set_dims(DSizes<Index, InRank>& _in_dims, DSizes<Index, OutRank>& _out_dims) {
    //  set_in_dims(_in_dims);
    //  set_out_dims(_out_dims);
    //}

    //void set_dims(LayerBase<Scalar>& layer) {
    //  if (are_dims_unset(layer.get_out_dims())) {
    //    set_dims(layer.get_out_dims(), layer.get_out_dims());
    //  }
    //}

    //void set_bias_dims(DSizes<Index, 1>& _bias_dims) { bias_dims = _bias_dims; }
    //void set_weight_dims(DSizes<Index, InRank>& _weight_dims) { weight_dims = _weight_dims; }


  protected:
    virtual ~LayerBase() = default;
    //DSizes<Index, InRank> in_dims, weight_dims;
    //DSizes<Index, OutRank> out_dims; 
    //DSizes<Index, 1> bias_dims;

    //bool are_dims_unset(std::vector<Index>& dims) { return in_dims.size() == 0 || dims[0] != in_dims[0]; }

  };
}