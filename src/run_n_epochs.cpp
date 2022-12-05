#include <iostream>
#include <vector>
#include <chrono>

#include "mlp.h"
#include "layers.h"
#include "read_data.h"

void run_mlp(std::vector< std::vector<double> > dataset, std::vector<double> y_true, int epochs);

// Show time increase as the number of epochs increases (iris dataset)
int main()
{
    // Read data from csv file
    ReadData read_data;
    read_data.create_dataset("data/iris.csv");

    // Set epochs
    int epochs = 20501;

    for (int ep = 500; ep < epochs; ep += 500){
        auto start = std::chrono::high_resolution_clock::now();

        run_mlp(read_data.get_my_data(), read_data.get_y_true(), ep);

        auto end = std::chrono::high_resolution_clock::now();
        double time = double(std::chrono::duration_cast <std::chrono::nanoseconds> (end-start).count());

        std::cout << "Execiton time for " << ep << " #epochs: " << time*1e-9 << std::endl;
    }
}

void run_mlp(std::vector< std::vector<double> > dataset, std::vector<double> y_true, int epochs)
{
    // Create mlp and set learning rate
    MLP mlp(dataset[0], true);
    mlp.set_learning_rate(0.2);

    int epoch = epochs;

    for (int e = 0; e < epoch; e++){
        for (int iter = 0; iter < dataset.size(); iter++){
            mlp.insert_sample(dataset[iter]);
            mlp.feed_forward(y_true[iter]);
            mlp.backpropagate(y_true[iter]);
        }
    }
}
