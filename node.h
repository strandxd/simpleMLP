#ifndef NODE_H
#define NODE_H

#include <vector>
#include <map>

class Node
{
    public:
        Node();
        Node(int idx, int output_weight_dim, bool bias_node=false, 
             double weight_init=0.5);

        // void activate_neuron(Layers &prev_layer);
        // Activations & calculations
        double sigmoid(double z);
        double sigmoid_derivative();
        void output_gradient(const int y_true);
        // void calc_output_gradient(const int y_true, const double out_err);

        // Set functions
        void set_output_value(double value) { output_value = value; }
        void set_output_value_bias() { output_value = 1.0; }
        void set_gradient(double val) { gradient = val; }
        
        // double random_weight() { return rand() / double(RAND_MAX); }

        // Get functions
        double get_output_value() { return output_value; }
        int get_curr_node_idx() { return curr_node_idx; }
        double get_gradient() { return gradient; }
        double get_loss() { return loss; }

        // Variables
        std::vector<double> output_weights; // weights for given perceptron. n vector
    
    private:
        double output_value; // Post activation value (just value in input lay)
        int curr_node_idx;
        double gradient;
        double loss;
};

#endif