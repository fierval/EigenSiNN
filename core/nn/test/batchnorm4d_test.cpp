#include <gtest/gtest.h>
#include <layers/batchnorm.hpp>
#include <ops/comparisons.hpp>

#include "include/commondata4d.hpp"
#include <layers\input.hpp>

using namespace EigenSinn;

namespace EigenSinnTest {
  class Batchnorm4dTest : public ::testing::Test {
  protected:
    void SetUp() override {

      cd.init();

      input = cd.convInput;
      loss = cd.batchNormLoss;
 
      gamma.resize(cd.dims[1]);
      beta.resize(cd.dims[1]);

      gamma.setValues({ 1., 2., 3., });
      beta = gamma * 0.1f;

      dinput.resize(cd.dims);
      output.resize(cd.dims);

      dinput.setValues({{{{-1.62950921, -0.69685334, -1.07382250, -1.76897717},
        {1.44790602, -1.49780560, -0.99879819, -1.40539515},
        {-0.59869730, -1.75040984, 1.25020134, -0.54927766},
        {0.91399211, -0.03358459, 1.29372990, 0.97778517}},

        {{ 3.21753860, -0.24326260, 0.98930210, -1.50716734},
        {-2.70493531, 2.52641654, 0.07768558, 1.48139572},
        {2.48122144, -0.72776258, 0.54139984, -2.82583928},
        {2.55554152, -0.20547122, 0.48839360, -0.03436335}},

        {{ 2.36492753, -4.52022934, 1.54314220, 0.75145990},
        {5.36521959, -2.28342438, -4.04580355, -1.20251715},
        {-4.27943039, -0.09987716, 1.00285757, -3.86846089},
        {3.66660929, -3.28293324, 0.94030267, 1.43936074}} },


        {{{ 1.86587000, 1.44176006, 1.62471294, -0.84876907},
        {0.89161289, 1.58880150, -1.49005330, 0.36842105},
        {-1.14859426, -0.78623319, -0.84980124, 1.20028925},
        {0.30062473, 0.91308540, -0.66597581, 1.71376503}},

        {{-0.48154697, 3.33144355, -1.46413648, 0.93390024},
        {-3.49391031, -3.27043295, -1.02778876, -1.13896823},
        {-1.66763973, -2.87063575, 2.92993975, 2.30749345},
        {2.51019764, -2.43451667, 0.48391661, -0.75740886}},

        {{ 4.18856192, -0.36916953, -2.26634550, -1.65659082},
        {4.37967062, 2.50316525, -3.79573894, -3.24426031},
        {3.41850448, 0.17727301, 2.41783452, -4.78536987},
        {-1.54826808, 4.95449162, -0.45484051, 2.58988047}}}});


      output.setValues({{{{-1.94571459, 1.44399858, -0.24312836, 0.09616741},
        {0.65938169, 0.60123307, -1.84200013, 1.06049955},
        {1.25588965, -0.36815381, -1.20263660, 0.07573818},
        {-1.01097441, 1.54873741, 0.22084165, -0.75563234}},

        {{ 0.14706215, -0.43545246, -1.02424788, 1.75614321},
        {-0.16912912, 3.47980428, 1.04790425, -1.55692101},
        {-0.94253904, -0.70965290, -0.09473468, 2.47096252},
        {1.11784589, -1.36432660, 0.43597293, 2.82975006}},

        {{ 3.20402884, 2.21821475, -4.38445377, -1.68158627},
        {2.10510349, -4.90361881, -0.04076374, 0.28052032},
        {1.27199268, 2.10043478, 4.47037983, -3.18100882},
        {-0.29413205, 4.65371466, -1.31607628, 2.29818177}} },


        {{{ 0.16357000, 1.02254784, -0.16685420, -1.44303179},
        {-0.67857045, -1.17160249, 0.64923638, -1.11255872},
        {0.55908424, 1.02274215, 0.19666116, 1.35757518},
        {0.28002760, 0.15972172, 1.45175838, 1.31544781}},

        {{-2.69912815, 2.65669250, 2.39568090, 3.65121031},
        {-0.87118751, -2.73704314, 2.86401629, -0.79330522},
        {-3.13175750, 1.31847310, -0.16828628, -2.80537176},
        {-1.77413309, 2.21832490, -2.82414508, 2.11151695}},

        {{ 4.04090929, -0.68238389, 2.90811658, -0.54438430},
        {-1.37870765, 1.01579809, -0.22508341, 1.50086093},
        {-5.88650417, -3.58296895, 4.43207932, 4.80590010},
        {-5.06263542, 3.31696224, -0.30335921, -1.55553436}} }});

      }

     //void TearDown() override {}

      DeviceTensor<float, 4> input, loss, dinput, output;
      Tensor<float, 1, RowMajor> beta, gamma;
      const float eps = 1e-5, momentum = 0.9;

      CommonData4d<ThreadPoolDevice> cd;

      // channel first
      const float prec = 1e-5;
    };

    TEST_F(Batchnorm4dTest, Backward) {

      Input<float> input_layer;
      input_layer.set_input(input);

      BatchNormalizationLayer<float, 4> bn(3);
      bn.init(beta, gamma);
      bn.forward(input_layer.get_output());
      bn.backward(input_layer.get_output(), loss.raw());

      DeviceTensor<float, 1> dbeta(cd.dims[1]), dgamma(cd.dims[1]);
      dbeta.setValues({ 15.55493259, 17.75424004, 14.48464108 });
      dgamma.setValues({ 0.19619252, -0.61846262,  2.08176923 });

      EXPECT_TRUE(is_elementwise_approx_eq(dinput, bn.get_loss_by_input_derivative(), 1e-5));
      EXPECT_TRUE(is_elementwise_approx_eq(dbeta, bn.get_loss_by_bias_derivative(), 2e-6));
      EXPECT_TRUE(is_elementwise_approx_eq(dgamma, bn.get_loss_by_weights_derivative(), 4e-6));
    }

  }