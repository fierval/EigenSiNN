#include "input.hpp"
#include "linear.hpp"
#include "Random.h"
#include <iostream>

using namespace std;

void test_fully_connected() {
  
  const int linear_layer_out = 3, batch_size = 2, linear_layer_in = 4;
  MatrixXd input_matrix;
  input_matrix.resize(batch_size, linear_layer_in);
  input_matrix << 
    1, 2, 3, 4,
    5, 6, 7, 8;

  //initialization constants
  RNG rng(1);
  double mu = 0;
  double sigma = 0.01;

  InputLayer input(input_matrix);
  Linear linear(input.batch_size(), input.input_vector_dim(), linear_layer_out);

  linear.init(rng, mu, sigma, true);

  MatrixXd final_grad = MatrixXd::Ones(input.batch_size(), linear_layer_out);
  linear.forward(input.get_layer());
  linear.backward(input.get_layer(), final_grad);

  cout << "X: " << endl;
  cout << input.get_layer() << endl << endl;

  cout << "Weights: " << endl;
  cout << linear.get_weights() << endl << endl;

  cout << "Y:" << endl;
  cout << linear.get_weights() << endl << endl;

  cout << "dL/dX:" << endl;
  cout << linear.get_loss_by_input_derivative() << endl << endl;

  cout << "dL/dW:" << endl;
  cout << linear.get_loss_by_weights_derivative() << endl << endl;
}

int main(int argc, char* argv[]) {

  test_fully_connected();
  return 0;
}