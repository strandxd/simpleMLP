#ifndef READ_DATA_H
#define READ_DATA_H

#include <vector>
#include <string>

class ReadData
{
    public:
        ReadData();
        void create_dataset(std::string filename);
        std::vector< std::vector<double> > get_my_data() { return my_data; }
        std::vector<double> get_y_true() { return y_true; }

    private:
        std::vector< std::vector<double> > my_data;
        std::vector<double> y_true;

};

#endif