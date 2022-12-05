#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include "read_data.h"


ReadData::ReadData() {}

/**
 * Read csv file and insert it into a matrix.
 * Read csv file code from: https://java2blog.com/read-csv-file-in-cpp/.
 * 
 * @param filename: name of file (including .csv)
*/
void ReadData::create_dataset(std::string filename)
{
    // Start code from: https://java2blog.com/read-csv-file-in-cpp/
    std::string fname = filename;
    
    std::vector< std::vector<std::string> > content;
    std::vector<std::string> row;
    std::string line, word;
    
    std::fstream file (fname, std::ios::in);
    if(file.is_open()){
        while(getline(file, line)){
            row.clear();
    
            std::stringstream str(line);
    
            while(getline(str, word, ','))
            row.push_back(word);
            content.push_back(row);
        }
    }
    else{
        std::cout<<"Could not open the file\n";
    }
    // End code from: https://java2blog.com/read-csv-file-in-cpp/


    // Convert data into matrix
    my_data.resize(content.size(), std::vector<double>(content[0].size()-1));
    y_true.resize(content.size());
    
    for(int row = 0; row < content.size(); row++){
        for(int col = 0; col < content[row].size(); col++){
            if (col == content[row].size()-1){ // last column = prediction
                y_true[row] = std::stoi(content[row][col]);
            }
            else{
                my_data[row][col] = std::stof(content[row][col]);
            }
        }
    }
}
