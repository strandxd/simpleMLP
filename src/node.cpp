#include <iostream>
#include <math.h>
#include <vector>
#include <assert.h>
#include <random>

#include "node.h"


/// @brief Empty constructor (needed for map construction)
Node::Node() {}

/**
 * Constructing node in layer with randomly initialized weights. Additionally sets output 
 * value for bias node to 1.
 * Node gets added to a map of <nodeId, nodeObject>.
 * 
 * NOTE: Could manually exchange random weight initialization with weight init (set to 0.5 if not
 * specified). 
 * 
 * @param idx: index of neuron.
 * @param output_weight_dim: number of nodes (not incl bias) in next layer.
 * @param weight_init: value of initial weight.
*/
Node::Node(int idx, int output_weight_dim, bool bias_node, double weight_init, 
           bool custom_weight_init)
{
    curr_node_idx = idx;
    // NBNB: Fix this a bit cleaner. Dont make array then add numbers
    output_weights.resize(output_weight_dim);

    // Initialize weights
    if (custom_weight_init){ // Default weight_init = 0.5
        for (int w = 0; w < output_weight_dim; w++){
            output_weights[w] = weight_init;
        }
    }
    else{
        for (int w = 0; w < output_weight_dim; w++){
            output_weights[w] = rand() / double(RAND_MAX);
        }
    }

    // Initialize output value for bias node
    if (bias_node){
        set_output_value_bias();
    }
}

/// @brief Deconstructor
Node::~Node()
{
    output_weights.clear();
    output_value = 0;
    curr_node_idx = 0;
    gradient = 0.0;
}

/**
 * Calculates gradient for output layer node.
 * Derivative of loss function:
 * (output_value - y_true) * output_value * (1 - outputvalue), i.e.:
 * (sigmoid(z) - y_true) * sigmoid(z) * (1 - sigmoid(z))
 * 
 * @param y_true: true target value.
*/
void Node::output_gradient(const int y_true)
{
    gradient = (output_value - y_true) * sigmoid_derivative();
}

/**
 * Squared error loss function (1/2 to make derivative cleaner).
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
*/
double Node::sigmoid_derivative()
{
    return output_value * (1 - output_value);
}