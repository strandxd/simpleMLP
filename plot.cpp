#include <iostream>
#include <chrono>

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1e9; i++){
        std::cout << "";
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time = double(std::chrono::duration_cast <std::chrono::nanoseconds> (end-start).count());

    std::cout << "Execiton time " << time*1e-9 << " seconds" << std::endl;
}


