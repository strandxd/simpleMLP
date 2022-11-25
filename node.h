#ifndef NODE_H
#define NODE_H

#include <vector>
#include <map>
#include <cstdlib>

class Node
{
    public:
        Node();
        Node(int idx, int output_weight_dim, bool bias_node=false, 
             double weight_init=0.5);

        // void activate_neuron(Layers &prev_layer);
        double sigmoid(double z);
        double sigmoid_derivative();

        void calc_output_gradient(const int y_true);

        void set_output_value(double value) { output_value = value; }
        void set_output_value_bias() { output_value = 1.0; }
        double get_output_value() { return output_value; }
        double random_weight() { return rand() / double(RAND_MAX); }

        // Variables
        std::vector<double> output_weights; // weights for given perceptron. n vector
        double output_value; // Post activation value (just value in input lay)
        int curr_node_idx;
        double gradient;
        // std::map<int, Layers> new_l;
        // std::map<int, Layers> new_l;
};

#endif