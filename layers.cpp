#include <iostream>
#include <vector>
#include <map>
#include <assert.h>
#include <math.h>

#include "layers.h"

Layers::Layers()
{

}

Layers::Layers(int layer_index, int num_nodes, int next_layer_dim)
{
    output_weight_dim = next_layer_dim;
    total_nodes = num_nodes + 1; // Add bias node
    layer_idx = layer_index;
    learning_rate = 0.1;

    // Bias node is the last iteration (node = total_nodes-1)
    for (int node = 0; node < total_nodes; node++){ 
        if (node == total_nodes-1){ // Create bias node with out_value=1.0
            map_of_nodes[node] = Node(node, next_layer_dim, true);
        }
        else{
            map_of_nodes[node] = Node(node, next_layer_dim);
        }
    }
}

void Layers::calc_hidden_gradients(Layers &right_layer)
{
    double sum_delta_hidden = 0.0;

    // Loop through every node in current layer (incl bias)
    for (int curr_n = 0; curr_n < total_nodes; curr_n++){

        // Loop through every node in layer to the right (not bias)
        for (int right_n = 0; right_n < right_layer.total_nodes-1; right_n++){
            sum_delta_hidden += 
            map_of_nodes[curr_n].output_weights[right_n] * 
            right_layer.map_of_nodes[right_n].gradient;
        }

        // Set gradient of current node
        map_of_nodes[curr_n].gradient = 
        sum_delta_hidden * map_of_nodes[curr_n].sigmoid_derivative();
    }
}

void Layers::update_weights(Layers &left_layer)
{
    // double delta_weight = 0.0;
    // Loop through every node in current layer (not bias)
    for (int curr_n = 0; curr_n < total_nodes-1; curr_n++){
        Node &curr_node = map_of_nodes[curr_n];
        double delta_weight = 0.0;
        // Loop through every node in left layer (incl bias)
        for (int left_n = 0; left_n < left_layer.total_nodes; left_n++){
            Node &left_node = left_layer.map_of_nodes[left_n];

            double delta_weight = 
            learning_rate * left_node.output_value * curr_node.gradient;

            left_node.output_weights[curr_n] += delta_weight; // Or is it -=??
        }
    }
}

/**
 * Activation of nodes
*/
void Layers::activate_nodes(Layers &prev_layer)
{
    // Inside current layer, has access to prev layer
    int current_node_id; // Store current node id

    // Loop through nodes in current layer, not bias
    for (int n_curr = 0; n_curr < total_nodes-1; n_curr++){
        double z = 0.0;
        current_node_id = map_of_nodes[n_curr].curr_node_idx;

        // Loop through all nodes (incl bias) in previous layer and add sum of: 
        // output_weights[current_node_id] * output_value
        for (int n_prev = 0; n_prev < prev_layer.total_nodes; n_prev++){
            Node &prev_node = prev_layer.map_of_nodes[n_prev];
            z += prev_node.output_weights[n_curr] * prev_node.output_value;
        }
        // Activate the current node
        map_of_nodes[n_curr].output_value = map_of_nodes[n_curr].sigmoid(z);
    }
}



