#include <iostream>
#include <vector>

#include "mlp.h"
#include "layers.h"
#include "read_data.h"

/**Runs mlp on iris dataset. Main showcase of the neural network.
 * Pretty decent results with {1 layer, 4 nodes, 5-6 epochs}. Displays the networks ability to 
 * seperate data
*/
int main()
{
    bool show_gradient;
    std::string temp_gradient;
    int epoch;

    // Read data from csv file.
    // NB: Need to be inside /bin to run file
    ReadData read_data;
    read_data.create_dataset("../data/iris.csv"); // Iris dataset

    // Create mlp
    MLP mlp(read_data.get_my_data()[0]);
    mlp.set_learning_rate(0.3);

    while (temp_gradient != "y" && temp_gradient != "n"){
        std::cout << "Display gradients (clutters output a bit)[y/n]?: ";
        std::cin >> temp_gradient;
    }
    show_gradient = (temp_gradient == "y") ? true : false;
    std::cout << "Number of epochs (one epoch = one iteration over every sample)?: ";
    std::cin >> epoch;

    int iter = 1; // Counter

    // Loop through e epochs
    for (int e = 0; e < epoch; e++){
        // For every epoch, loop through entire dataset (i.e. 1 iteration = 1 sample)
        for (int row = 0; row < read_data.get_my_data().size(); row++){
            std::cout << "\n*********************************" << std::endl;
            std::cout << "EPOCH #" << e+1 << ". TOTAL ITERATION #" << iter << std::endl;

            // One iteration contains following steps:
            mlp.insert_sample(read_data.get_my_data()[row]);
            mlp.feed_forward(read_data.get_y_true()[row]);
            mlp.print_output_values(show_gradient);
            mlp.print_results(read_data.get_y_true()[row]);
            mlp.backpropagate(read_data.get_y_true()[row]);

            iter++;
            std::cout << "*********************************" << std::endl;
            std::cout << '\n';
        }
    }
}
