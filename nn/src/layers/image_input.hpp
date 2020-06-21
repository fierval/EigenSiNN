#pragma once

#include "layer_base.hpp"
#include "ops/convolutions.hpp"

#define MAX_IMAGE_SIZE 1e6
#define MAX_IMAGE_BATCH 1e6
#define IMAGE_CHANNELS 4

namespace EigenSinn {
  class ImageInput : LayerBase {
  
  public:
    ImageInput(ConvTensor& _layer_content)
    : image(_layer_content) {

      assert(image.dimension(0) > 0 && image.dimension(0) <= MAX_IMAGE_BATCH && image.dimension(3) > 0 && image.dimension(3) <= IMAGE_CHANNELS);
    }

    const ConvTensor& get_layer() {
      return image;
    }

  private:
    ConvTensor image;
  };
}