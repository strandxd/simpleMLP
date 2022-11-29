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
    MLP::learning_rate = 0.1;

    int epoch = 10;
    int iter = 1;

    for (int e = 0; e < epoch; e++){
        for (int row = 0; row < read_data.get_my_data().size(); row++){
            std::cout << "EPOCH #" << e+1 << ". TOTAL ITERATION #" << iter << std::endl;
            // y_true should have as many inputs as rows in dset

            // One iteration contains following steps:
            mlp.insert_sample(read_data.get_my_data()[row]);
            mlp.feed_forward();
            mlp.print_results(read_data.get_y_true()[row]);
            mlp.backpropagate(read_data.get_y_true()[row]);

            iter++;
        }
    }
}
