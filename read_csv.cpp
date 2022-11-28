#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>


class MyData
{
    public: 
        MyData() {}

        std::vector< std::vector<double> > my_data;
        std::vector<double> new_y_true;
    
        void create_dset()
        {
            std::string fname = "iris.csv";
            // std::string fname = "iris_std.csv";
            // std::cout<<"Enter the file name: ";
            // std::cin>>fname;
            
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

            my_data.resize(content.size(), std::vector<double>(content[0].size()-1));
            new_y_true.resize(content.size());
            
            for(int i = 0; i < content.size(); i++){
                
                for(int j = 0; j < content[i].size(); j++){
                    if (j == content[i].size()-1){ // last column = prediction
                        new_y_true[i] = std::stoi(content[i][j]);
                    }
                    else{
                        my_data[i][j] = std::stof(content[i][j]);
                    }
                }
            }

            // for (int row=0; row < my_data.size(); row++){
            //     for (int col = 0; col < my_data[0].size(); col++){
            //         std::cout << my_data[row][col] << " ";
            //     }
            //     std::cout << std::endl;
            // }
        }
};
