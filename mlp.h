#ifndef MLP_H
#define MLP_H

#include <iostream>
#include <vector>
#include <map>

#include "layers.h"


class MLP
{
    public:
        // Constructors
        MLP();
        MLP(std::vector<double> &input_row);

        void feed_forward(std::vector<double> &input_data);
        void backpropagate(const int y_true);

        void print_results(const int y_true);
        void print_output_values();

        // Vars
        std::map<int, Layers> map_of_layers;
        double output_error;
};


#endif