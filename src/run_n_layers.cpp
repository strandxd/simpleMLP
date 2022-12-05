#include <iostream>
#include <vector>
#include <chrono>

#include "mlp.h"
#include "layers.h"
#include "read_data.h"

void run_mlp(std::vector< std::vector<double> > dataset, std::vector<double> y_true, int n_layers);

// Show time increase as the number of layers increases (iris dataset)
int main()
{
    // Read data from csv file
    // NB: Need to be inside /bin to run file
    ReadData read_data;
    read_data.create_dataset("../data/iris.csv");

    int max_layers = 5000;

    for (int num_layer = 20; num_layer < max_layers; num_layer += 200){
        auto start = std::chrono::high_resolution_clock::now();

        run_mlp(read_data.get_my_data(), read_data.get_y_true(), num_layer);

        auto end = std::chrono::high_resolution_clock::now();
        double time = double(std::chrono::duration_cast <std::chrono::nanoseconds> 
                     (end-start).count());

        std::cout << "Execiton time for " << num_layer << " #layers: " << time*1e-9 << std::endl;
    }
}

void run_mlp(std::vector< std::vector<double> > dataset, std::vector<double> y_true, int n_layers)
{
    // Create mlp and set learning rate
    MLP mlp(dataset[0], n_layers);
    mlp.set_learning_rate(0.2);

    for (int iter = 0; iter < dataset.size(); iter++){
        mlp.insert_sample(dataset[iter]);
        mlp.feed_forward(y_true[iter]);
        mlp.backpropagate(y_true[iter]);
    }
}

