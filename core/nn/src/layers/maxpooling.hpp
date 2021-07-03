#pragma once

#include "layer_base.hpp"
#include <ops/conversions.hpp>
#include <device/device_helpers.hpp>
#include <device/device_maxpool.hpp>
#include <helpers/conv_params_bag.hpp>
#include <limits>

#include <onnx/op_defs.h>

#ifdef __CUDACC__
#include "cudnn/cudnn_pooling.hpp"
#endif

using namespace  Eigen;

namespace EigenSinn {

  // NHWC format
  // For Linear (fully connected) layers: (N, C)
  // N - batch size
  // C - number of channels (1 for fully connected layers)
  template <typename Scalar, Index Rank = 4, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class MaxPooling : public LayerBase<Scalar, Device_> {
  public:

    MaxPooling(const DSizes<Index, Rank / 2>& _extents, Index _stride, Padding2D _padding = { 0, 0 }, int _dilation = 1)
      : LayerBase<Scalar, Device_>(maxpool_op)
      , extents(_extents)
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
        // need to prepend the right values to mimic convolutional kernel (batch, channels)
        dims = x.dimensions();

        DSizes<Index, Rank> kernel_dims;

        kernel_dims[0] = dims[0];
        kernel_dims[1] = dims[1];
        kernel_dims[2] = extents[0];
        kernel_dims[3] = extents[1];

        params = std::make_shared<MaxPoolParams<Rank>>(dims, kernel_dims, padding, stride, dilation, false);

        // cache input and output dimensions
        out_dims = params->output_dims();

        // allocate output tensors
        layer_output = DeviceTensor<Scalar, Rank, Device_, Layout>(out_dims);
        mask = DeviceTensor<Index, Rank, Device_, Layout>(out_dims);

        layer_gradient = DeviceTensor<Scalar, Rank, Device_, Layout>(dims);
      }

      int w_offset = -padding.first;
      int h_offset = -padding.second;

#ifndef __CUDACC__
      if (!std::is_same<Device_, GpuDevice>::value) {
        // parallelize maxpooling
        std::for_each(std::execution::par_unseq, params->output_range.begin(), params->output_range.end(), [&](auto out_index) {

          DSizes<Index, Rank> offsets = from_flat_dim<Index, Rank, Layout>(out_dims, out_index);
          Index b = offsets[0], c = offsets[1], h = offsets[2], w = offsets[3];

          Scalar max_val = std::numeric_limits<Scalar>::lowest();
          Index max_idx = -1;

          for (Index kernel_h = 0; kernel_h < params->dilated_kernel_height; kernel_h += dilation) {
            for (Index kernel_w = 0; kernel_w < params->dilated_kernel_width; kernel_w += dilation) {
              Index cur_h = h_offset + h * stride + kernel_h;
              Index cur_w = w_offset + w * stride + kernel_w;

              Scalar val;

              if (cur_h >= 0 && cur_h < dims[2] && cur_w >= 0 && cur_w < dims[3]) {
                val = (*x)(b, c, cur_h, cur_w);
                max_idx = (val > max_val) ? to_flat_dim<Index, 4, Layout>(dims, { b, c, cur_h, cur_w }) : max_idx;
                max_val = (val > max_val) ? val : max_val;
              }
            }
          }
          layer_output->data()[out_index] = max_val;
          mask->data()[out_index] = max_idx;
          });
      }
      else {
#else
      if (is_cudnn && !cudnn_pooling) {
        cudnn_pooling = std::make_shared<CudnnPooling<Scalar, Rank>>(dims, out_dims, CUDNN_POOLING_MAX, *params);
      }

      if (is_cudnn) {
        cudnn_pooling->forward(x->data(), layer_output->data());
        return;
      }

      static int block(BLOCK_SIZE * BLOCK_SIZE);
      static int grid(getGridSize(out_dims.TotalSize(), BLOCK_SIZE * BLOCK_SIZE));
      auto stream = mask.device().stream();

      maxpool_forward_kernel<Scalar, Layout> << <grid, block, 0, stream >> >
        (h_offset, w_offset, params->dilated_kernel_height, params->dilated_kernel_width, stride, dilation, *x, *layer_output, *mask);

      cudaDeviceSynchronize();
#endif
#ifndef __CUDACC__
      }
#endif
    }

  // for derivations
  void backward(LayerBase<Scalar, Device_>& prev_layer, PtrTensorAdapter<Scalar, Device_> next_layer_grad) override {

    DeviceTensor<Scalar, Rank, Device_, Layout> x(next_layer_grad);

    layer_gradient.setZero();

#ifndef __CUDACC__
    if (!std::is_same<Device_, GpuDevice>::value) {
      std::for_each(std::execution::par_unseq, params->output_range.begin(), params->output_range.end(), [&](auto out_index) {
        Index idx = mask->data()[out_index];

        std::lock_guard<std::mutex> lck(mtx);
        layer_gradient->data()[idx] += x->data()[out_index];
        });
    }
    else {
#else

    if (is_cudnn) {
      cudnn_pooling->backward(x->data(), layer_gradient->data());
      return;
    }

    static int block(BLOCK_SIZE * BLOCK_SIZE);
    static int grid(getGridSize(dims.TotalSize(), BLOCK_SIZE * BLOCK_SIZE));
    auto stream = mask.device().stream();

    maxpool_backward_kernel<Scalar, Layout> << <grid, block, 0, stream >> >
      (-padding.first, -padding.second, params->dilated_kernel_height, params->dilated_kernel_width, stride, dilation, *x, *mask, *layer_gradient);

    cudaDeviceSynchronize();
#endif
#ifndef __CUDACC__
    }
#endif
  }

  PtrTensorAdapter<Scalar, Device_> get_output() override {
    return layer_output.raw();
  }

  PtrTensorAdapter<Scalar, Device_> get_loss_by_input_derivative() {
    return layer_gradient.raw();
  }

#ifdef __CUDACC__
  std::shared_ptr<CudnnPooling<Scalar, Rank>> cudnn_pooling;
#endif

  inline void set_cudnn(bool _is_cudnn) {

    if (Rank < 4) { return; }
    assert(!_is_cudnn || Rank > 2 && Layout == RowMajor);

    is_cudnn = _is_cudnn;
  }

  const std::string add_onnx_node(EigenModel& model, const std::string& input_name) override {

    // https://github.com/onnx/onnx/blob/v1.9.0/docs/Operators.md#MaxPool
    onnx::NodeProto* node = model.add_graph_node(maxpool_op, input_name);

    const std::string out_name = node->output().Get(0);

    params->create_onnx_attributes(node);

    // save rank, not part of ONNX but necessary for loading
    model.add_attr(node, "rank", Rank);


    // return output to pass as input to next node in graph
    return out_name;
  }

  const std::vector<Index> onnx_out_dims() override {
    return layer_output.vec_dims();
  }

  MaxPoolParams<Rank>& get_maxpool_params() { return *params; }

  private:
    DeviceTensor<Scalar, Rank, Device_, Layout> layer_output, layer_gradient;
    DeviceTensor<Index, Rank, Device_, Layout> mask;

    DSizes<Index, Rank / 2> extents;
    DSizes<Index, Rank> out_dims;
    DSizes<Index, Rank> dims;

    const int stride;
    const Padding2D padding;
    const int dilation;

    std::shared_ptr<MaxPoolParams<Rank>> params;

    // concurrency
    std::mutex mtx;
  };

}