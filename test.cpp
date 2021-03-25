#include <iostream>
#include "util/ground_truth.hpp"

using namespace std;
using namespace MultipleHNSW;

int main() {
    char *dataset = "sift";
    char *method_name = "hnsw";
    vector <pair<double, double>> time_recall_l; //time: us, recall:0-1
    vector<int> ef_search_l = {10, 20, 30};
    int M = 16;
    int ef_construction = 100;
    double indexing_time = 0.3;
    long max_memory = 100;
    pair<double, double> p1(0.1, 0.99);
    time_recall_l.push_back(p1);
    save_recall(dataset, method_name, time_recall_l, ef_search_l, M, ef_construction, indexing_time, max_memory);

    return 0;
}