#include "input.hpp"
#include "linear.hpp"


void test_fully_connected() {
  
  const int linear_layer_out = 3, batch_size = 2, linear_layer_in = 2;
  MatrixXd input_matrix;
  input_matrix.resize(batch_size, linear_layer_in);
  input_matrix << 
    1, 4,
    2, 3;

  InputLayer input(input_matrix);
  Linear linear(input.batch_size(), input.input_vector_dim(), linear_layer_out);

  MatrixXd final_grad = MatrixXd::Ones(input.batch_size(), linear_layer_out);
  linear.forward(input.get_layer());
  linear.backward(final_grad);
}

int main(int argc, char* argv[]) {

  test_fully_connected();
  return 0;
}