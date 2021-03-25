#include <iostream>
#include <vector>

namespace MultipleHNSW {

    class AlgorithmInterface {
    public:
        virtual void insert(const int *item, int label) = 0;

        virtual std::vector<int> search_knn(const int * query, int k)=0;

        virtual ~AlgorithmInterface() {
        }

    };

}
