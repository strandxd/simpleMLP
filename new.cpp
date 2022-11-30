#include <map>
#include <iostream>


int main(){
    std::map<int, int> new_map;

    new_map[1] = 2;
    new_map[2] = 3;
    new_map[3] = 4;

    // for (auto itr = new_map.begin(); itr != new_map.end(); itr++){
    //     std::cout << itr->second << " ";
    // }
    // std::cout << std::endl;

    // for (auto itr = new_map.begin(); itr != new_map.end(); itr++){
    //     std::cout << itr->second << " ";
    // }

    std::cout << new_map.at(2);
}