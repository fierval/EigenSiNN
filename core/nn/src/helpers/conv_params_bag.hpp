#pragma once

#include "ops/opsbase.hpp"

using namespace Eigen;

namespace EigenSinn {

  template<Index Rank>
  class ConvolutionParams {

  public:
    // A bag of convolution parameters and cached values
    ConvolutionParams(const DSizes<Index, Rank>& _in_dims, const DSizes<Index, Rank> _kernel_dims,
      const Padding2D& _padding, const int _stride, const int _dilation, const bool _is_transposed)
      : input_dims(_in_dims)
      , kernel_dims(_kernel_dims)
      , padding(_padding)
      , stride(_stride)
      , dilation(_dilation)
      , is_transposed(_is_transposed) {

      get_output_dimensions();
      set_kernel_positions();
      set_batches_range();

      dilated_kernel_height = dilation * (kernel_dims[2] - 1) + 1;
      dilated_kernel_width = dilation * (kernel_dims[3] - 1) + 1;

      dilated_kernel_dims = kernel_dims;
      dilated_kernel_dims[2] = dilated_kernel_height;
      dilated_kernel_dims[3] = dilated_kernel_width;
    }

    inline void get_output_dimensions() {

      if (!is_transposed) {
        assert(kernel_dims[(int)ImageDims::channel] == input_dims[(int)ImageDims::channel]);
      }
      else {
        assert(kernel_dims[(int)ImageDims::batch] == input_dims[(int)ImageDims::channel]);
      }

      assert(kernel_dims[(int)ImageDims::height] > 0 && kernel_dims[(int)ImageDims::width] > 0);

      Index pad_height = 2 * padding.first;
      Index pad_width = 2 * padding.second;

      out_dims[(int)ImageDims::batch] = input_dims[(int)ImageDims::batch];
      out_dims[(int)ImageDims::channel] = !is_transposed ? kernel_dims[(int)ImageDims::batch] : kernel_dims[(int)ImageDims::channel];

      if (!is_transposed) {
        // see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        out_dims[(int)ImageDims::height] = (input_dims[(int)ImageDims::height] + pad_height - dilation * (kernel_dims[(int)ImageDims::height] - 1) - 1) / stride + 1;
        out_dims[(int)ImageDims::width] = (input_dims[(int)ImageDims::width] + pad_width - dilation * (kernel_dims[(int)ImageDims::width] - 1) - 1) / stride + 1;
      }
      // see https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
      else {
        out_dims[(int)ImageDims::height] = (input_dims[(int)ImageDims::height] - 1) * stride - pad_height + dilation * (kernel_dims[(int)ImageDims::height] - 1) + 1;
        out_dims[(int)ImageDims::width] = (input_dims[(int)ImageDims::width] - 1) * stride - pad_width + dilation * (kernel_dims[(int)ImageDims::width] - 1) + 1;
      }
    }

    // Defines how to apply the kernel to images in case we are doing it on the CPU
    inline void set_kernel_positions() {

      h_im_range.resize(output_dims()[(int)ImageDims::height]);
      w_im_range.resize(output_dims()[(int)ImageDims::width]);

      std::iota(h_im_range.begin(), h_im_range.end(), 0);
      std::iota(w_im_range.begin(), w_im_range.end(), 0);

      std::transform(h_im_range.begin(), h_im_range.end(), h_im_range.begin(), [=](auto i) {return i * stride - padding.first; });
      std::transform(w_im_range.begin(), w_im_range.end(), w_im_range.begin(), [=](auto i) {return i * stride - padding.second; });
    }

    // for col2im loop.
    // the "batches" here are applications of the kernel
    // not the training batches
    inline void set_batches_range() {

      const DSizes<Index, Rank>& dims = output_dims();
      col_batches.resize(dims[(int)ImageDims::height] * dims[(int)ImageDims::width]);
      std::iota(col_batches.begin(), col_batches.end(), 0);
    }

    inline const DSizes<Index, Rank>& orig_dims() const {
      return is_transposed ? out_dims : input_dims;
    }

    inline const DSizes<Index, Rank>& output_dims() const {
      return is_transposed ? input_dims : out_dims;
    }

    // this can happen if we switch from "train" to "test" mode
    // so our batch may change
    inline bool check(const DSizes<Index, Rank> new_input_dims) {
      return new_input_dims[(int)ImageDims::batch] != input_dims[(int)ImageDims::batch];
    }

    const DSizes<Index, Rank> kernel_dims;
    DSizes<Index, Rank> dilated_kernel_dims;

    const Padding2D padding;
    const int stride;
    const int dilation;

    Index dilated_kernel_height;
    Index dilated_kernel_width;
    const bool is_transposed;

    std::vector<long> h_im_range, w_im_range, col_batches;

  private:
    // dimensions depend on convolution type (transposed/regular)
    // what is "input" for regular is "output" for transposed
    DSizes<Index, Rank> out_dims;
    DSizes<Index, Rank> input_dims;

  };
} // namespace