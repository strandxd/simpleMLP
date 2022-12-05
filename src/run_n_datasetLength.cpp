#include <vector>
#include <iostream>
#include <random>
#include <chrono>

#include "mlp.h"

std::vector< std::vector<double> > generate_matrix(int max_rows);
std::vector<int> generate_y_true(int max_rows);

// Show time increase as the length of dataset increases (samples(/rows), randomly generated data)
int main()
{
    int row_number = 300000; // 300,000

    for (int max_rows = 100; max_rows < row_number; max_rows += 10000){
        // Generate random dataset
        std::vector< std::vector<double> > sample_matrix = generate_matrix(max_rows);
        std::vector<int> sample_y_true = generate_y_true(max_rows);

        // Initialize mlp class
        MLP mlp(sample_matrix[0], true);
        mlp.set_learning_rate(0.2);

        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < sample_matrix.size(); iter++){
            mlp.insert_sample(sample_matrix[iter]);
            mlp.feed_forward(sample_y_true[iter]);
            mlp.backpropagate(sample_y_true[iter]);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time = double(std::chrono::duration_cast <std::chrono::nanoseconds> 
                     (end-start).count());

        std::cout << "Execiton time for " << max_rows << " rows: " << time*1e-9 << std::endl;
    }
}

// Generate y true values
std::vector<int> generate_y_true(int max_rows)
{
    std::vector<int> simulated_y_true(max_rows);

    for (int row = 0; row < max_rows; row++){
        simulated_y_true[row] = rand() % 2; // Random number 1 or 0
    }

    return simulated_y_true;
}

// Generate matrix with random numbers
std::vector< std::vector<double> > generate_matrix(int max_rows)
{
    int cols = 20;
    std::vector< std::vector<double> > simulated_matrix(max_rows, std::vector<double>(cols));

    for (int row = 0; row < max_rows; row++){
        for (int col = 0; col < cols; col++){
            simulated_matrix[row][col] = rand() / double(RAND_MAX);
        }
    }
    
    return simulated_matrix;
}