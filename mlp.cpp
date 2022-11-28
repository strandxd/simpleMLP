#include <iostream>
#include <vector>
#include <map>
#include <math.h>

#include "mlp.h"

MLP::MLP()
{

}

/**
 * 
 * @param input_row: First row of input matrix (to know input layer dim)
*/
MLP::MLP(std::vector<double> &input_row)
{
    int input_dimension = input_row.size(); // Features dim + bias
    int n_layers;
    int n_nodes_layer;
    int output_dimension;

    // Create the network
    std::map<int, int> layer_node_setup; // key: layer#, value: n_nodes_layer

    std::cout << "How many hidden layers?: ";
    std::cin >> n_layers;
    n_layers += 2; // Add input and output layer

    for (int i = 0; i < n_layers; i++){
        if (i == 0){ // Input layer
            layer_node_setup[i] = input_dimension;
        }
        else if (i == n_layers-1){ // Output layer
            layer_node_setup[i] = 1;
        }
        else{ // Hidden layers
            std::cout << "How many neurons in hidden layer # " << i << "?: ";
            std::cin >> n_nodes_layer;
            layer_node_setup[i] = n_nodes_layer;
        }
    }

    // This creates bias node for output layer, but whatever
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



/**
 * 
 * @param input_data: new sample row
*/
void MLP::feed_forward(std::vector<double> &input_data)
{
    // Add values to input layer (bias is last node).
    int total_input_nodes = map_of_layers[0].total_nodes;
    for (int i = 0; i < total_input_nodes; i++){
        if (i == total_input_nodes-1){ // Bias node
            map_of_layers[0].map_of_nodes[i].set_output_value_bias();
        }
        else{
            map_of_layers[0].map_of_nodes[i].set_output_value(input_data[i]);
        }
    }

    // PRINT INPUT VALS. DELETE
    std::cout << "INPUT VALS: " << std::endl;
    for (int i = 0; i < map_of_layers[0].map_of_nodes.size(); i++){
        std::cout << map_of_layers[0].map_of_nodes[i].get_output_value() << " ";
    }
    std::cout << std::endl;
    // PRINT INPUT VALS. DELETE

    // Loop through the rest of the layers
    for (int layer = 1; layer < map_of_layers.size(); layer++){
        Layers &left_layer = map_of_layers[layer-1];
        Layers &current_layer = map_of_layers[layer];

        current_layer.activate_nodes(left_layer);

        // Loop through nodes in layer except bias node (last node)
        for (int node = 0; node < map_of_layers[layer].total_nodes - 1; node++){
            map_of_layers[layer].activate_nodes(left_layer);
        }
    }
}


void MLP::backpropagate(const int y_true)
{
    // Calculate error (y_true - output_value)
    Layers &output_layer = map_of_layers[map_of_layers.rbegin()->first];
    output_error = output_layer.map_of_nodes[0].get_output_value() - y_true;
    output_error = pow(output_error, 2); // square the error

    // Log loss (binary crossentropy)
    // double out_val = output_layer.map_of_nodes[0].get_output_value();
    // output_error = -((y_true * log(out_val)) + ((1 - y_true) * log(1 - out_val))); 
    // Log loss (binary crossentropy)

    // Calculate outputlayer gradient
    output_layer.map_of_nodes[0].output_gradient(y_true);

    // Calculate hidden layer gradient (from last hidden layer down to (including) 1st hidden layer)
    for (int layer = map_of_layers.size()-2; layer > 0; layer--){
        Layers &right_layer = map_of_layers[layer + 1];

        map_of_layers[layer].hidden_gradients(right_layer);
    }
    // From output layer to first hidden layer, update weights
    for (int layer = map_of_layers.size() - 1; layer > 0; layer--){
        Layers &left_layer = map_of_layers[layer - 1];

        map_of_layers[layer].update_weights(left_layer);
    }
}

void MLP::print_results(const int y_true)
{
    Layers &output_layer = map_of_layers[map_of_layers.size()-1];

    double predicted_value = output_layer.map_of_nodes[0].get_output_value();

    std::cout << "True value: " << y_true << std::endl;;
    std::cout << "Predicted value: " << predicted_value << std::endl;
    std::cout << "Loss: " << output_layer.map_of_nodes[0].get_loss() << std::endl;
    std::cout << "\n";
    
    // int hidden_l = map_of_layers.size() - 2;
    // std::cout << "Weights: ";
    // for (int node = 0; node < map_of_layers[hidden_l].map_of_nodes.size(); node++){
    //     for (int w = 0; 
    //     w < map_of_layers[hidden_l].map_of_nodes[node].output_weights.size(); w++){
    //         std::cout << map_of_layers[hidden_l].map_of_nodes[node].output_weights[w] << " ";
    //     }
    // }
    // std::cout << '\n' << std::endl;
}

void MLP::print_output_values()
{
    for (int layer = 0; layer < map_of_layers.size(); layer++){
        std::cout << "Layer #" << layer << std::endl;
        for (int node = 0; node < map_of_layers[layer].map_of_nodes.size(); node++){
            double node_out = map_of_layers[layer].map_of_nodes[node].get_output_value();
            std::cout << "Neuron #" << node << " . Value: " << node_out << '\n';
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