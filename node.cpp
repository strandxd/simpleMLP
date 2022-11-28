#include <iostream>
#include <math.h>
#include <vector>
#include <assert.h>
#include <random>

#include "node.h"

/// @brief Empty constructor
Node::Node() {}

/**
 * Constructing node in layer with randomly initialized weight dimension. Additionally sets output 
 * value for bias node to 1.
 * Node gets added to a map of <nodeNumer, nodeObject>.
 * 
 * NOTE: Could manually exchange random weight initialization with weight init (set to 0.5 if not
 * specified). 
 * 
 * @param idx: index of neuron.
 * @param output_weight_dim: number of nodes (not incl bias) in next layer.
 * @param weight_init: value of initial weight.
*/
Node::Node(int idx, int output_weight_dim, bool bias_node, double weight_init)
{
    curr_node_idx = idx;
    // Fix this a bit cleaner. Dont make array then add numbers
    output_weights.resize(output_weight_dim);

    // Initialize weights
    for (int w = 0; w < output_weight_dim; w++){
        output_weights[w] = rand() / double(RAND_MAX);
    }

    // Initialize output value for bias node
    if (bias_node){
        set_output_value_bias();
    }

    // // Print weights (DELETE)
    // std::cout << "Initial weights : \n";
    // for (int i =0; i < output_weights.size(); i++){
    //     std::cout << output_weights[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "Output value: " << output_value << std::endl;
}

/**
 * Calculates gradient for output layer node.
 * 
 * @param y_true: true value of sample target.
*/
void Node::output_gradient(const int y_true)
{
    loss = output_value - y_true;
    // double delta_output = out_err - y_true;
    gradient = loss * sigmoid_derivative();
}

/**
 * Activation of node. Based on inputs from connected neurons in previous layer,
 * calculate the activation of node.
 * 
 * @param z: inner product (weight dot value) of previous connected layer.
 * @return: sigmoid function value.
*/
double Node::sigmoid(double z)
{
    return 1 / (1 + exp(-z));
}

/**
 * Derivative of sigmoid function. Used for calculating gradients.
*/
double Node::sigmoid_derivative()
{
    return output_value * (1 - output_value);
}