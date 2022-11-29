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
        MLP(std::vector<double> &input_row, bool default_init = false);

        void insert_sample(std::vector<double> &input_data);
        void feed_forward();
        void backpropagate(const int y_true);

        void print_results(const int y_true);
        void print_output_values();

        // Vars
        std::map<int, Layers> map_of_layers;
        double output_error;
        static double learning_rate;
};


#endif