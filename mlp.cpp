#include <iostream>
#include <vector>
#include <map>
#include <math.h>

#include "mlp.h"

double MLP::learning_rate = 0.1; // Set learning rate

/// @brief Empty constructor
MLP::MLP() {}

/**
 * Constructing network object. Initializes map of user defined number of layers, which in turn 
 * initializes map of user defined number of nodes, i.e. entire network gets initialized when 
 * calling this constructor.
 * 
 * Refering to part 1 and 2 in comments below.
 * Part 1: Create a map containing layer number and layer of nodes. Done to keep track of number 
 * of nodes in next layer (to know weight dimension). Map consists of <layerId, numberOfNodes>.
 * 
 * Part 2: Loop through layers and add <Layers> object to map.
 * 
 * @param input_row: First row of input matrix (to define input layer dimension).
 * @param num_layers: Number of layers in network (mainly used for test cases).
 * @param default_init: Default initiation of network (n_layers=1, n_nodes=4). Mainly for testing.
*/
MLP::MLP(std::vector<double> const &input_row, int num_layers, bool default_init)
{
    int input_dimension = input_row.size(); // Features dim + bias
    int n_layers;
    int n_nodes_layer;
    int output_dimension;

    // Part 1: Define temporary map for storing network structure
    std::map<int, int> layer_node_setup; // key: layer#, value: n_nodes_layer(excl. bias)

    if (num_layers > 0){
        n_layers = num_layers;
    }
    else if (!default_init){
        std::cout << "How many hidden layers?: ";
        std::cin >> n_layers;
    }
    else{
        n_layers = 1;
    }
    n_layers += 2; // Add input and output layer

    for (int i = 0; i < n_layers; i++){
        if (i == 0){ // Input layer
            layer_node_setup[i] = input_dimension;
        }
        else if (i == n_layers-1){ // Output layer
            layer_node_setup[i] = 1;
        }
        else{ // Hidden layers
            if (num_layers > 0){ // Default initiation of nodes = 10
                n_nodes_layer = 10;
            }
            else if (!default_init){
                std::cout << "How many nodes in hidden layer # " << i << "?: ";
                std::cin >> n_nodes_layer;
            }
            else{
                n_nodes_layer = 4;
            }
            layer_node_setup[i] = n_nodes_layer;
        }
    }

    // Part 2: Fill in class map <map_of_layers>
    // Note: This creates an unnecessary bias node for output layer (num nodes+1 called in <Layers>)
    for (int layer = 0; layer < n_layers; layer++){
        if (layer == n_layers-1){ // Output layer has no "next_layer_dim"
            map_of_layers[layer] = Layers(layer, 1, 0);
        }
        else{
            map_of_layers[layer] = Layers(layer, 
                layer_node_setup[layer], layer_node_setup[layer+1]);
        }
    }
}

MLP::~MLP()
{
    map_of_layers.clear();
}


/**
 * Fill input layer with values from an inserted sample row.
 * 
 * @param input_data: sample data row
*/
void MLP::insert_sample(std::vector<double> const &input_data)
{
    int total_input_nodes = map_of_layers[0].get_total_nodes();
    for (int i = 0; i < total_input_nodes; i++){
        if (i == total_input_nodes-1){ // Bias node
            continue; // Already sat output value when initializing nodes
        }
        else{
            map_of_layers[0].map_of_nodes[i].set_output_value(input_data[i]);
        }
    }

    // // PRINT INPUT VALS. DELETE
    // std::cout << "INPUT VALS: " << std::endl;
    // for (int i = 0; i < map_of_layers[0].map_of_nodes.size(); i++){
    //     std::cout << map_of_layers[0].map_of_nodes[i].get_output_value() << " ";
    // }
    // std::cout << std::endl;
    // // PRINT INPUT VALS. DELETE
}


/**
 * Feed forward process of neural network. From first hidden layer to output layer, calculate 
 * activatons of individual nodes in each layer.
*/
void MLP::feed_forward()
{
    // Loop through 1st hidden layer to ouput layer
    for (int layer = 1; layer < map_of_layers.size(); layer++){
        Layers &left_layer = map_of_layers[layer-1];
        Layers &current_layer = map_of_layers[layer];

        current_layer.activate_nodes(left_layer);

        // Loop through nodes in layer (excl. bias node)
        for (int node = 0; node < map_of_layers[layer].get_total_nodes() - 1; node++){
            map_of_layers[layer].activate_nodes(left_layer);
        }
    }
}

/**
 * Backpropagation process of neural network. 
 * 
 * Step 1: Calculate gradient of output layer.
 * Step 2: Calculate gradients of hidden layer.
 * Step 3: Update each weight.
 * 
 * @param y_true: true target value.
*/
void MLP::backpropagate(const int y_true)
{
    // Step 1: Output layer gradient
    Layers &output_layer = map_of_layers[map_of_layers.rbegin()->first];
    output_layer.map_of_nodes[0].output_gradient(y_true);

    // Step 2: Hidden layer gradients
    // Calculate hidden layer gradient (from last hidden layer down to (including) 1st hidden layer)
    for (int layer = map_of_layers.size()-2; layer > 0; layer--){
        Layers &right_layer = map_of_layers[layer + 1];

        map_of_layers[layer].hidden_gradients(right_layer);
    }

    // Step 3: Update weights
    // From output layer to first hidden layer, update weights
    for (int layer = map_of_layers.size() - 1; layer > 0; layer--){
        Layers &left_layer = map_of_layers[layer - 1];

        map_of_layers[layer].update_weights(left_layer, learning_rate);
    }
}

/**
 * Print output for single iteration (one sample).
 * 
 * @param y_true: true target value.
*/
void MLP::print_results(const int y_true)
{
    Layers &output_layer = map_of_layers[map_of_layers.rbegin()->first];

    double predicted_value = output_layer.map_of_nodes[0].get_output_value();

    std::cout << "True value: " << y_true << std::endl;;
    std::cout << "Predicted value: " << predicted_value << std::endl;
    // std::cout << "Loss: " << output_layer.map_of_nodes[0].get_loss() << std::endl;
}

/**
 * Prints output values from each node in every layer.
*/
void MLP::print_output_values()
{
    for (int layer = 0; layer < map_of_layers.size(); layer++){
        if (layer == 0){
            std::cout << "Input layer: " << std::endl;
        }
        else if (layer == map_of_layers.size()-1){
            std::cout << "Output layer: " << std::endl;
        }
        else{
            std::cout << "Hidden layer #" << layer << std::endl;      
        }
        for (int node = 0; node < map_of_layers[layer].map_of_nodes.size(); node++){
            double node_out = map_of_layers[layer].map_of_nodes[node].get_output_value();
            std::cout << "Node #" << node << ". Value: " << node_out << std::endl;
        }   
        std::cout << std::endl;
    }
}

// void MLP::print_outputs()
// {
//     int lay_c = 0;
//     for (auto l_itr = map_of_layers.begin(); l_itr != map_of_layers.end(); l_itr++){
//         std::cout << "Layer #" << lay_c << std::endl;
//         std::cout << "Output: ";
//         for (auto n_itr = l_itr->second.map_of_nodes.begin(); 
//         n_itr != l_itr->second.map_of_nodes.end(); n_itr++){
//             std::cout << n_itr->second.output_value << " ";
//         }
//         lay_c++;
//         std::cout << std::endl;
//     }
// }