#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <vector>
#include <map>

#include "node.h"

class Layers
{
    public:
        // Constructors
        Layers();
        Layers(int layer_index, int num_nodes, int next_layer_dim);

        void activate_nodes(Layers &prev_layer);
        void hidden_gradients(Layers &right_layer);
        void update_weights(Layers &left_layer);

        // Pub vars?
        std::map<int, Node> map_of_nodes;
        int output_weight_dim;
        
        // Private vars?
        int layer_idx;
        int total_nodes;
        
        
        double learning_rate;
};

#endif