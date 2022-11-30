#include <iostream>
#include <vector>
#include <map>
#include <assert.h>
#include <math.h>

#include "layers.h"

/**
 * General note: Just to avoid confusion when referencing layers in the following section.
 * We think of the neural network as going left to right through forward propagation.
 * When referencing layers, current layer is obviously the one we are in, and layers to 
 * the left and right of this layer are related to this forward propagation perspective. 
 * This then also applies when we loop through the layer in reverse (backpropagation).
*/

// double Layers::learning_rate = 0.1; // Set learning rate


/// @brief Empty constructor
Layers::Layers() {}

/**
 * Constructing layer object in network. Initializes layer with a user specifed number of nodes. 
 * Layer gets added to a map of <layerNumber, layerObject>.
 * 
 * @param layer_index: layer number.
 * @param num_nodes: number of nodes in layer (excluding bias node).
 * @param next_layer_dim: number of nodes in next layer (excluding bias node).
*/
Layers::Layers(int layer_index, int num_nodes, int next_layer_dim)
{
    set_output_weight_dim(next_layer_dim);
    set_total_nodes(num_nodes+1);
    set_layer_idx(layer_index);
    // learning_rate = 0.1;

    // Bias node is the last iteration (node = total_nodes-1)
    for (int node = 0; node < get_total_nodes(); node++){ 
        if (node == get_total_nodes()-1){ // Create bias node with "output value=1.0"
            map_of_nodes[node] = Node(node, get_output_weight_dim(), true);
        }
        else{
            map_of_nodes[node] = Node(node, get_output_weight_dim());
        }
    }
}

Layers::~Layers()
{
    map_of_nodes.clear();
    set_output_weight_dim(0);
    set_layer_idx(0);
    set_total_nodes(0);
}

/**
 * Related to backpropagation.
 * Calculate and set the gradient of hidden layer.
 * 
 * For every node in the current layer (incl. bias), loop through every node in layer to the right
 * (excl. bias). Sum up all (outgoing weights * gradient) corresponding to the node it relates to.
 * Multiply sum with the derivative of the activation function and update gradient of current node 
 * in current layer.
 * 
 * @param right_layer: reference of layer object to the right of current layer.
*/
void Layers::hidden_gradients(Layers &right_layer)
{
    double sum_delta_hidden;
    double gradient_value;

    // Loop through every node in current layer (incl. bias)
    for (int curr_n = 0; curr_n < get_total_nodes(); curr_n++){

        // Loop through every node in layer to the right (excl. bias)
        for (int right_n = 0; right_n < right_layer.get_total_nodes()-1; right_n++){
            sum_delta_hidden += 
            map_of_nodes[curr_n].output_weights[right_n] * 
            right_layer.map_of_nodes[right_n].get_gradient();
        }

        // Set gradient of current node
        gradient_value = sum_delta_hidden * map_of_nodes[curr_n].sigmoid_derivative();
        map_of_nodes[curr_n].set_gradient(gradient_value);
    }
}

/**
 * Related to backpropagation.
 * Update weights of outgoing weights in the layer to the left of current layer.
 * 
 * For every node in the current layer (excl. bias), loop thorugh every node in the layer to the
 * left (incl. bias). Calculate the weight change and add this to weights connecting left layer to
 * correct node in current layer.
 * 
 * @param left_layer: reference of layer object to the left of current layer.
*/
void Layers::update_weights(Layers &left_layer, double learning_rate)
{
    double delta_weight = 0.0;
    // Loop through every node in current layer (not bias)
    for (int curr_n = 0; curr_n < get_total_nodes()-1; curr_n++){
        Node &curr_node = map_of_nodes[curr_n];

        // Loop through every node in left layer (incl bias)
        for (int left_n = 0; left_n < left_layer.get_total_nodes(); left_n++){
            Node &left_node = left_layer.map_of_nodes[left_n];

            double delta_weight = 
            -(learning_rate * left_node.get_output_value() * curr_node.get_gradient());

            left_node.output_weights[curr_n] += delta_weight; // Or is it -=??
        }
    }
}

/**
 * Activation of nodes
*/
void Layers::activate_nodes(Layers &left_layer)
{
    // Inside current layer, has access to prev layer
    int current_node_id;
    double out_value;

    // Loop through nodes in current layer (excl. bias)
    for (int n_curr = 0; n_curr < get_total_nodes()-1; n_curr++){
        double z = 0.0;
        current_node_id = map_of_nodes[n_curr].get_curr_node_idx();

        // Loop through all nodes (incl bias) in previous layer and add sum of: 
        for (int n_prev = 0; n_prev < left_layer.get_total_nodes(); n_prev++){
            Node &prev_node = left_layer.map_of_nodes[n_prev];
            z += prev_node.output_weights[n_curr] * prev_node.get_output_value();
        }
        // Activate the current node
        out_value = map_of_nodes[n_curr].sigmoid(z);
        map_of_nodes[n_curr].set_output_value(out_value);
    }
}
