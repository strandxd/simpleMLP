#include <iostream>
#include <math.h>
#include <vector>
#include <assert.h>

#include "node.h"

Node::Node()
{

}

/**
 * 
 * @param idx: index of neuron
 * @param output_weight_dim: number of neurons (not inkl bias) in next layer
 * @param weight_init: value of initial weight
*/
Node::Node(int idx, int output_weight_dim, bool bias_node, double weight_init)
{
    curr_node_idx = idx;
    // Fix this a bit cleaner. Dont make array then add numbers
    output_weights.resize(output_weight_dim);
    for (int w = 0; w < output_weight_dim; w++){
        output_weights[w] = random_weight();
    }

    if (bias_node){
        set_output_value_bias();
    }
    // Print weights (DELETE)
    // std::cout << "WEIGHTS: \n";
    // for (int i =0; i < output_weights.size(); i++){
    //     std::cout << output_weights[i] << " ";
    // }
    // std::cout << std::endl;
}

/**
 * Calculates gradient for output layer node
*/
void Node::calc_output_gradient(const int y_true)
{
    double delta_output = y_true - output_value;
    gradient = delta_output * sigmoid_derivative();
}

/**
 * Activation of node. Based on inputs from connected neurons in previous layer,
 * calculate the activation of node.
 * 
 * @param z: inner product of (weight * value) of previous connected layer.
 * @return: sigmoid function value.
*/
double Node::sigmoid(double z)
{
    return 1 / (1 + exp(-z));
}

double Node::sigmoid_derivative()
{
    return output_value * (1 - output_value);
}