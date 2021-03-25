#include <iostream>
#include <vector>
#include <algorithm> // std::move_backward
#include <random> // std::default_random_engine
#include <chrono> // std::chrono::system_clock

int main (int argc, char* argv[])
{
    std::vector<int> v;

    for (int i = 0; i < 10; ++i) {
        v.push_back (i);
    }

    // obtain a time-based seed:
    unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count ();
    std::shuffle (v.begin (), v.end (), std::default_random_engine (seed));

    for (auto& it : v) {
        std::cout << it << " ";
    }

    std::cout << "\n";

    return 0;
}