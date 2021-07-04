#include <iostream>
#include <gtest/gtest.h>
#include <layers/tanh.hpp>
#include "include/commondata2d.hpp"
#include "ops/comparisons.hpp"
#include <layers/input.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {

  class TanhGpu : public ::testing::Test {

  protected:
    void SetUp() override {

      output.resize(cd.dims);
      dinput.resize(cd.dims);

      cd.init();

      output.setValues({{ 0.70301139, -0.90233523,  0.55932426,  0.36756060,  0.39080179,
                  -0.60449260,  0.10666516,  0.98704076},
                 {-0.92004395,  0.67083871, -0.94689846,  0.96894515, -0.17930010,
                   0.78161395, -0.04075922, -0.31478795},
                 {-0.22843744, -0.80275923, -0.93291336, -0.46897596, -0.60890055,
                   0.24278317, -0.08900084,  0.53964549}});

      dinput.setValues({{1.85131580e-01, 7.55941123e-02, 6.02954924e-01, 5.37525475e-01,
                  7.35940337e-01, 3.26052368e-01, 5.96295893e-01, 1.55762827e-04},
                 {3.24201807e-02, 7.56537989e-02, 2.44991910e-02, 9.85141844e-03,
                  8.91914546e-01, 2.05327615e-01, 2.71000038e-03, 2.94971168e-01},
                 {3.33942175e-01, 2.82350898e-01, 7.04460293e-02, 3.63111079e-01,
                  1.02910064e-01, 4.72531915e-01, 1.70583978e-01, 3.74054343e-01}});
    }

    CommonData2d<GpuDevice> cd;
    DeviceTensor<float, 2, GpuDevice> output, dinput;

  };

  TEST_F(TanhGpu, Backward) {

    Input<float, GpuDevice> input;
    input.set_input(cd.linearInput);

    EigenSinn::Tanh<float, 2, GpuDevice> tanh;
    tanh.init();
    tanh.forward(input);
    tanh.backward(input, cd.linearLoss.raw());
 
    EXPECT_TRUE(is_elementwise_approx_eq(tanh.get_output(), output));
    EXPECT_TRUE(is_elementwise_approx_eq(tanh.get_loss_by_input_derivative(), dinput));
  }
}