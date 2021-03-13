#pragma once

#include "layer_base.hpp"
#include <ops/maxpoolingops.hpp>
#include <ops/conversions.hpp>
#include <device/device_helpers.hpp>
#include <limits>

using namespace  Eigen;

namespace EigenSinn {

  // NHWC format
  // For Linear (fully connected) layers: (N, C)
  // N - batch size
  // C - number of channels (1 for fully connected layers)
  template <typename Scalar, Index Rank = 4, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class MaxPooling : public LayerBase<Scalar, Device_> {
  public:

    MaxPooling(const DSizes<Index, Rank/2>& _extents, Index _stride, Padding2D _padding = { 0, 0 }, int _dilation = 1)
      : extents(_extents)
      , stride(_stride)
      , padding(_padding)
      , dilation(_dilation) {
      
      static_assert(Rank == 4, "MaxPooling is implemented only for Rank == 4");

    }

    void init() override {
      
    }

    // Implementation from darknet: https://github.com/pjreddie/darknet/blob/9a4b19c4158b064a164e34a83ec8a16401580850/src/maxpool_layer.c
    void forward(LayerBase<Scalar, Device_>& prev_layer) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> x(prev_layer.get_output());

      if (!params || params->check(x.dimensions())) {
        // dimensions represented by vector for Rank = 2 or Rank = 4
        // need to prepend the right values to mimic convolutional kernel
        DSizes<Index, Rank> dims = x.dimensions();
        DSizes<Index, Rank> kernel_dims = dims;
        kernel_dims[0] = dims[0];
        kernel_dims[1] = dims[1];

        params = std::make_shared<ConvolutionParams<Rank>>(dims, kernel_dims, padding, stride, dilation, false);
      }

      int w_offset = -padding.first;
      int h_offset = -padding.second;

      DSizes<Index, Rank> out_dims = params->output_dims();
      DSizes<Index, Rank> dims = params->orig_dims();

      for (Index b = 0; b < out_dims[0]; ++b) {
        for (Index c = 0; c < out_dims[1]; c++) {
          for (Index h = 0; h < out_dims[2]; ++h) {
            for (Index w = 0; w < out_dims[3]; ++w) {

              int out_index = to_flat_dim<Index, 4, Layout>(out_dims, {b, c, h, w});
              Scalar max_val = std::numeric_limits<Scalar>::lowest();
              Index max_idx = -1;

              for (Index kernel_h = 0; kernel_h < params->dilated_kernel_height; kernel_h += dilation) {
                for (Index kernel_w = 0; kernel_w < params->dilated_kernel_width; kernel_w += dilation) {
                  Index cur_h = h_offset + h * stride + kernel_h;
                  Index cur_w = w_offset + w * stride + kernel_w;

                  Scalar val;

                  if (cur_h >= 0 && cur_h < dims[2] && cur_w >= 0 && cur_w < dims[3]) {
                    val = (*x)(b, c, cur_h, cur_w);
                  }
                  else {
                    val = std::numeric_limits<Scalar>::lowest();
                  }

                  max_idx = (val > max_val) ? to_flat_dim<Index, 4, Layout>(dims, { b, c, cur_h, cur_w }) : max_idx;
                  max_val = (val > max_val) ? val : max_val;
                }
              }
              layer_output->data()[out_index] = max_val;
              mask->data()[out_index] = max_idx;
            }
          }
        }
      }
    }

    // for derivations
    void backward(LayerBase<Scalar, Device_>& prev_layer, PtrTensorAdapter<Scalar, Device_> next_layer_grad) override {

      DeviceTensor<Scalar, Rank, Device_, Layout> x(next_layer_grad);

      for (Index i = 0; i < params->output_dims().TotalSize(); i++) {
        Index idx = mask->data()[i];
        layer_gradient->data()[idx] += x->data()[i];
      }
    }

    PtrTensorAdapter<Scalar, Device_> get_output() override {
      return layer_output.raw();
    }

    PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() {
      return layer_gradient.raw();
    }


  private:
    DeviceTensor<Scalar, Rank, Device_, Layout> layer_output, layer_gradient;
    DeviceTensor<Index, Rank, Device_, Layout> mask;

    DSizes<Index, Rank/2> extents;

    const int stride;
    const Padding2D padding;
    const int dilation;

    std::shared_ptr<ConvolutionParams<Rank>> params;
  };

}