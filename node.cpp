#include <iostream>
#include <math.h>
#include <vector>
#include <assert.h>
#include <random>

#include "node.h"

/// @brief Empty constructor (needed for map construction)
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
    // NBNB: Fix this a bit cleaner. Dont make array then add numbers
    output_weights.resize(output_weight_dim);

    // Initialize weights
    for (int w = 0; w < output_weight_dim; w++){
        output_weights[w] = rand() / double(RAND_MAX);
    }

    // Initialize output value for bias node
    if (bias_node){
        set_output_value_bias();
    }
}


Node::~Node()
{
    output_weights.clear();
    output_value = 0;
    curr_node_idx = 0;
    gradient = 0.0;
    error = 0.0;
}

/**
 * Calculates gradient for output layer node.
 * 
 * @param y_true: true target value.
*/
void Node::output_gradient(const int y_true)
{
    error = output_value - y_true;
    gradient = error * sigmoid_derivative();
}

/**
 * Squared error loss function (1/2 * SE just to make derivative cleaner).
 * 
 * @param y_true: true target value
*/
void Node::calculate_loss(const int y_true)
{
    loss = (1.0/2.0) * (pow((y_true - output_value), 2));
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
 * Derivative is: sigmoid(z) * (1 - sigmoid(z)), and as we know, sigmoid(z) = output_value.
 * 
 * NB: Not implemented yet, maybe do it! or delete this!
 * Note: The nodes are always activated before calling sigmoid_derivative. Although, to add a
 * safety mechanism, we initiate the output_value to -1 and check if the value is >= 0 (as it will 
 * always be using the sigmoid function. Setting >= to avoid round-off errors).
*/
double Node::sigmoid_derivative()
{
    return output_value * (1 - output_value);
}