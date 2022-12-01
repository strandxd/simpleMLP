#ifndef NODE_H
#define NODE_H

#include <vector>
#include <map>

class Node
{
    public:
        // Constructors & destructor
        Node();
        Node(int idx, int output_weight_dim, bool bias_node=false, 
             double weight_init=0.5);
        ~Node();

        // Activations & calculations
        double sigmoid(double z);
        double sigmoid_derivative();
        void output_gradient(const int y_true);

        // Set functions
        void set_output_value(double value) { output_value = value; }
        void set_output_value_bias() { output_value = 1.0; }
        void set_gradient(double val) { gradient = val; } const

        // Get functions
        double get_output_value() { return output_value; }
        int get_curr_node_idx() { return curr_node_idx; }
        const double get_gradient() { return gradient; }
        double get_error() { return error; }

        std::vector<double> output_weights; // weights for given node. n vector
    
    private:
        double output_value; // Post activation value
        int curr_node_idx;
        double gradient;
        double error;
};

#endif