#include <iostream>
#include <fstream>
#include "src/bruteforce.hpp"
#include "util/vecs_io.hpp"

using namespace std;
using namespace MultipleHNSW;

int main() {

    std::printf("load ground truth\n");
    int gnd_n_vec = 100;
    int gnd_vec_dim = 10;
    char* path = "../siftsmall/gnd.ivecs";
    int* gnd = MultipleHNSW::read_ivecs(gnd_n_vec, gnd_vec_dim, path);
//    std::printf("%d %d %d %d %d\n", gnd[0], gnd[1], gnd[2], gnd[3], gnd[4]);

    std::printf("load query\n");
    int query_n_vec = 100;
    int query_vec_dim = 128;
    path = "../siftsmall/query.bvecs";
    int* query = MultipleHNSW::read_bvecs(query_n_vec, query_vec_dim, path);
//    std::printf("%d %d %d %d %d\n", query[0], query[1], query[2], query[3], query[4]);

    std::printf("load base\n");
    int base_n_vec = 10000;
    int base_vec_dim = 128;
    path = "../siftsmall/base.bvecs";
    int* base = MultipleHNSW::read_bvecs(base_n_vec, base_vec_dim, path);
//    std::printf("%d %d %d %d %d\n", base[0], base[1], base[2], base[3], base[4]);

    BruteForceSearch bfs(base_vec_dim);

    for(int i=0;i<base_n_vec;i++){
        bfs.insert(base + base_vec_dim * i, i);
    }

    printf("querying\n");
    vector<vector<int>> test_gnd_l;
    for(int i=0;i<gnd_n_vec;i++){
        vector<int> test_gnd = bfs.search_knn(query + i * query_vec_dim, gnd_vec_dim);
        test_gnd_l.push_back(test_gnd);
    }
    bool equal = true;
    for(int i=0;i<gnd_n_vec;i++){
        for(int j=0;j<gnd_vec_dim;j++){
            if(test_gnd_l[i][j] != gnd[i * gnd_vec_dim + j]){
                equal = false;
            }
        }
    }

    if(equal){
        printf("equal to the ground truth\n");
    }else{
        printf("not equal to the ground truth\n");
    }

    return 0;
}