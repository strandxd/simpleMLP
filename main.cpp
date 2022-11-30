#include <iostream>
#include <vector>
#include <time.h>
#include <assert.h>
#include <fstream>
#include <string>
#include <sstream>

#include "mlp.h"
#include "layers.h"
#include "read_data.h"

int main()
{
    // Read data from csv file
    ReadData read_data;
    read_data.create_dataset("iris.csv");

    // Create 
    MLP mlp(read_data.get_my_data()[0]);
    mlp.set_learning_rate(0.3);

    int epoch = 100;
    int iter = 1;

    // Loop through e epochs
    for (int e = 0; e < epoch; e++){
        // For every epoch, loop through entire dataset (i.e. 1 iteration = 1 sample)
        for (int row = 0; row < read_data.get_my_data().size(); row++){
            std::cout << "\n*********************************" << std::endl;
            std::cout << "EPOCH #" << e+1 << ". TOTAL ITERATION #" << iter << std::endl;

            // One iteration contains following steps:
            mlp.insert_sample(read_data.get_my_data()[row]);
            mlp.feed_forward();
            mlp.print_output_values();
            mlp.print_results(read_data.get_y_true()[row]);
            mlp.backpropagate(read_data.get_y_true()[row]);

            iter++;
            std::cout << "*********************************" << std::endl;
            std::cout << '\n';
        }
    }
}
