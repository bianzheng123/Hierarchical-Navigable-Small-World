#include <iostream>
#include <fstream>
#include "src/hnsw.hpp"
#include "util/vecs_io.hpp"
#include "util/ground_truth.hpp"
#include "util/time_memory.hpp"

using namespace std;
using namespace MultipleHNSW;

int main() {
    int M = 40;
    int ef_construction = 100;
    int ef_search = 100;
    int random_seed = 100;
    std::printf("load ground truth\n");
    int gnd_n_vec = 100;
    int gnd_vec_dim = 10;
    char *path = "../data/siftsmall/gnd.ivecs";
    int *gnd = read_ivecs(gnd_n_vec, gnd_vec_dim, path);
//    std::printf("%d %d %d %d %d %d\n", gnd[0], gnd[1], gnd[2], gnd[3], gnd[4], gnd[10]);

    std::printf("load query\n");
    int query_n_vec = 100;
    int query_vec_dim = 128;
    path = "../data/siftsmall/query.bvecs";
    int *query = read_bvecs(query_n_vec, query_vec_dim, path);
//    std::printf("%d %d %d %d %d\n", query[0], query[1], query[2], query[3], query[4]);

    std::printf("load base\n");
    int base_n_vec = 10000;
    int base_vec_dim = 128;
    path = "../data/siftsmall/base.bvecs";
    int *base = read_bvecs(base_n_vec, base_vec_dim, path);
//    std::printf("%d %d %d %d %d\n", base[0], base[1], base[2], base[3], base[4]);

    HNSW hnsw(base_vec_dim, base_n_vec, M, ef_construction, ef_search, random_seed);

    size_t report_every = 1000;
    TimeRecord insert_record;

    for (int i = 0; i < base_n_vec; i++) {
        hnsw.insert(base + base_vec_dim * i, i);

        if (i % report_every == 0) {
            cout << i / (0.01 * base_n_vec) << " %, "
                 << 1e-6 * insert_record.get_elapsed_time_micro() << " s/iter" << " Mem: "
                 << get_current_RSS() / 1000000 << " Mb \n";
            insert_record.reset();
        }
    }

    printf("querying\n");
    vector <vector<int>> test_gnd_l;
    double single_query_time;
    TimeRecord query_record;
    for (int i = 0; i < gnd_n_vec; i++) {
        vector<int> test_gnd = hnsw.search_knn(query + i * query_vec_dim, gnd_vec_dim);
        test_gnd_l.push_back(test_gnd);
    }
    single_query_time = query_record.get_elapsed_time_micro() / query_n_vec * 1e-3;
    printf("self program, M %d, efConstruction %d, efSearch %d\n", M, ef_construction, ef_search);
    hnsw.test_graph();

    double recall = count_recall(gnd_n_vec, gnd_vec_dim, test_gnd_l, gnd);
    printf("average recall: %.3f, single query time %.1f ms\n", recall, single_query_time);
    printf("peak memory %d Mb\n", get_peak_RSS() / 1000000);
    return 0;
}