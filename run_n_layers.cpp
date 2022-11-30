#include <iostream>
#include <vector>
#include <chrono>

#include "mlp.h"
#include "layers.h"
#include "read_data.h"

void run_mlp(std::vector< std::vector<double> > dataset, std::vector<double> y_true, int n_layers);

int main()
{
    // Read data from csv file
    ReadData read_data;
    read_data.create_dataset("iris.csv");

    int max_layers = 1000;

    for (int num_layer = 1; num_layer < max_layers; num_layer += 50){
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
        // std::cout << "EPOCH #" << e+1 << ". TOTAL ITERATION #" << iter+1 << std::endl;
        // y_true should have as many inputs as rows in dset

        // One iteration contains following steps:
        mlp.insert_sample(dataset[iter]);
        mlp.feed_forward();
        // mlp.print_results(read_data.get_y_true()[iter]);
        mlp.backpropagate(y_true[iter]);
    }
}

