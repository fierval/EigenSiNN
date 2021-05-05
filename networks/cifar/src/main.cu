#include "main.hpp"

using namespace EigenSinn;
using namespace Eigen;

bool should_shuffle = true;
bool explore_dataset = false;
bool debug_init = false;

int main(int argc, char* argv[]) {

  size_t batch_size = 10, inference_batch_size = 10;
  int side = 32;
  int num_epochs = 5;
  int channels = 3;
  float learning_rate = 0.001;
  
  std::vector<std::string> classes = { "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
  int num_classes = classes.size();

  auto dataset = read_cifar_dataset();

  explore(dataset, explore_dataset);

  ImageContainer<ColMajor> next_images;
  LabelContainer next_labels;

  auto net = Cifar10<GpuDevice>(DSizes<Index, 4>{(Index)batch_size, channels, side, side}, num_classes, learning_rate);
  net.init();

  auto start = std::chrono::high_resolution_clock::now();
  auto start_step = std::chrono::high_resolution_clock::now();
  auto stop = std::chrono::high_resolution_clock::now();

  // Train
  for (int i = 0; i < num_epochs; i++) {

    start = std::chrono::high_resolution_clock::now();
    start_step = std::chrono::high_resolution_clock::now();
    
    shuffle<float, uint8_t>(dataset.training_images, dataset.training_labels, should_shuffle);

    // execute network step
    for (int step = 1; step <= dataset.training_images.size() / batch_size; step++) {

      DeviceTensor<float, 4, GpuDevice> batch_tensor(create_batch_tensor(dataset.training_images, step - 1, batch_size));
      DeviceTensor<uint8_t, 2, GpuDevice> label_tensor(create_2d_label_tensor<uint8_t>(dataset.training_labels, step - 1, batch_size, num_classes));

      // training step 
      net.step(batch_tensor, label_tensor);

      if (step % 1000 == 0) {
        stop = std::chrono::high_resolution_clock::now();
        std::cout << "Epoch: " << i + 1 
          << ". Step: " << step 
          << ". Loss: " << net.get_loss() 
          << ". Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start_step).count() / 1000. 
          << "." << std::endl;

        start_step = std::chrono::high_resolution_clock::now();
      }
    } 

    stop = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.;

    std::cout << "Epoch: " << i + 1 << ". Time: " << elapsed << " sec." << " Loss: " << net.get_loss() << "." << std::endl;
  } 
  
  // Test
  TestNetworkSingleBatch(dataset, net, num_classes, classes, inference_batch_size);
  return 0;
}