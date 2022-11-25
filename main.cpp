#include <iostream>
#include <vector>
#include <time.h>
#include <assert.h>
#include <fstream>
#include <string>
#include <sstream>

#include "mlp.h"
#include "layers.h"
#include "read_csv.cpp"

std::vector< std::vector<double> > create_dataset()
{
std::string fname;
std::cout<<"Enter the file name: ";
std::cin>>fname;
 
std::vector<std::vector<std::string>> content;
std::vector<std::string> row;
std::string line, word;
double num;
 
std::fstream file (fname, std::ios::in);
if(file.is_open())
{
while(getline(file, line))
{
row.clear();
 
std::stringstream str(line);
 
while(getline(str, word, ','))
row.push_back(word);
content.push_back(row);
}
}
else
std::cout<<"Could not open the file\n";

std::vector< std::vector<double> > new_m(content.size(), 
std::vector<double>(content[0].size()));
 
for(int i=0;i<content.size();i++){
    
    for(int j=0;j<content[i].size();j++){
        new_m[i][j] = std::stof(content[i][j]);
    }
}

for (int row=0; row < new_m.size(); row++){
    for (int col = 0; col < new_m[0].size(); col++){
        std::cout << new_m[row][col] << " ";
    }
    std::cout << std::endl;
}
 
return new_m;
}

std::vector< std::vector<double> > create_dset(int rows, int cols)
{
    srand (time(NULL));
    std::vector< std::vector<double> > dset(rows, std::vector<double>(cols));

    for (int row = 0; row < dset.size(); row++){
        for (int col = 0; col < dset[0].size(); col++){
            dset[row][col] = (rand() % 5);
        }
    }

    return dset;

}

std::vector <int> create_y_true(int rows)
{
    srand (time(NULL));
    std::vector<int> y_true(rows);
    for (int i = 0; i < rows; i++){
        y_true[i] = rand() % 2; // 0 or 1
    }
    return y_true;
}

void print_row(std::vector<double> vec)
{
    for (int i = 0; i < vec.size(); i++){
        std::cout << vec[i] << " ";
    }
}

int main()
{
    int rows = 100000;
    int cols = 20;
    // Create and print dset
    std::vector< std::vector<double> > new_dset = create_dset(rows, cols);
    std::vector<int> y_true = create_y_true(rows);
    // for (int row = 0; row < new_dset.size(); row++){
    //     for (int col = 0; col < new_dset[0].size(); col++){
    //         if (col == new_dset[0].size()-1){
    //             std::cout << new_dset[row][col] << " " << y_true[row];
    //         }
    //         else{
    //             std::cout << new_dset[row][col] << " ";
    //         }
            
    //     }
    //     std::cout << std::endl;
    // }

    MyData data_class;

    data_class.create_dset();

    MLP test_mlp(new_dset[0]);
    int epoch = 10;
    int new_epoch = 1;

    for (int e = 0; e < epoch; e++){
        int iter = 0;
        for (int row = 0; row < new_dset.size(); row++){
            std::cout << "ITERATION #" << iter << std::endl;
            // y_true should have as many inputs as rows in dset
            assert(y_true.size() == new_dset.size());

            test_mlp.feed_forward(new_dset[row]);
            test_mlp.backpropagate(y_true[row]);
            test_mlp.print_results(y_true[row]);
            // test_mlp.print_output_values();

            iter++;
        }
        new_epoch++;
    }


    // std::cout << "\nWeights hidden: \n";
    // int hidden = test_mlp.map_of_layers.size() - 2;
    // for (double weight : test_mlp.map_of_layers[0].map_of_nodes[2].output_weights){
    //     std::cout << weight << " ";
    // }


    // MLP new_mlp(new_dset[0]);

    // new_mlp.feed_forward(new_dset[0]);
    // std::cout << "PAssed forward\n";
    // new_mlp.print_results();
    // std::cout << "Passed print";

}
