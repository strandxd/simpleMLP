#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <vector>
#include <map>

#include "node.h"

class Layers
{
    public:
        // Constructors & destructor
        Layers();
        Layers(int layer_index, int num_nodes, int next_layer_dim);
        ~Layers();

        void activate_nodes(Layers &prev_layer);
        void hidden_gradients(Layers &right_layer);
        void update_weights(Layers &left_layer, double learning_rate);

        // Set functions
        void set_layer_idx(int value) { layer_idx = value; }
        void set_total_nodes(int value) { total_nodes = value; }
        void set_output_weight_dim(int value) { output_weight_dim = value; }

        // Get functions
        int get_layer_idx() { return layer_idx; } const
        int get_total_nodes() { return total_nodes; } const 
        int get_output_weight_dim() { return output_weight_dim; }

        std::map<int, Node> map_of_nodes;
        
        private:
            int layer_idx;
            int total_nodes;
            int output_weight_dim;
};

#endif