#ifndef MLP_H
#define MLP_H

#include <iostream>
#include <vector>
#include <map>

#include "layers.h"


class MLP
{
    public:
        // Constructors & destructor
        MLP();
        MLP(std::vector<double> const &input_row, int num_layers = 0, bool default_init = false);
        ~MLP();

        // Run network
        void insert_sample(std::vector<double> const &input_data);
        void feed_forward(const int y_true);
        void backpropagate(const int y_true);

        // Set functions
        void set_learning_rate(double value) { learning_rate = value; }

        // Print
        void print_results(const int y_true);
        void print_output_values(bool show_gradient = false);

        std::map<int, Layers> map_of_layers;

        private:
            static double learning_rate;
};


#endif