#include <vector>
#include <iostream>
#include <random>

int main()
{
    std::vector<double> n(10);
    std::cout << "HELLO";

    for (int i = 0; i < 10; i++){
        n[i] = (rand() % 2);
    }

    std::cout << "RANDOM NUM";

    for (int j = 0; j < n.size(); j++){
        std::cout << n[j] << " ";
    }
}