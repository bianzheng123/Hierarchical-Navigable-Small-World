#include <iostream>
#include <vector>
#include <unordered_set>
#include <fstream>

#if defined(_WIN32)

#error "Cannot make directory for windows OS."

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <sys/stat.h>
#include <sys/types.h>

#else
#error "Cannot make directory for an unknown OS."
#endif

namespace MultipleHNSW {

    std::vector<std::unordered_set<int>> change_gnd_type(int n_query, int topk, int *gnd) {
        std::vector<std::unordered_set<int>> res(n_query);
        for (int i = 0; i < n_query; i++) {
            for (int j = 0; j < topk; j++) {
                res[i].insert(gnd[i * topk + j]);
            }
        }
        return res;
    }

    double count_recall(int n_query, int topk, std::vector<std::vector<int>> test_gnd_l, int *gnd) {
        std::vector<std::unordered_set<int>> gnd_set = change_gnd_type(n_query, topk, gnd);
        std::vector<double> recall_l;
        double avg_recall = 0;

        for (int i = 0; i < n_query; i++) {
            int match = 0;
            int test_gnd_size = test_gnd_l[i].size();
            for (int j = 0; j < test_gnd_size; j++) {
                if (gnd_set[i].find(test_gnd_l[i][j]) != gnd_set[i].end()) { //find
                    match++;
                }
            }
            double recall = (double) match / topk;
            recall_l.push_back(recall);
            avg_recall += recall;
        }
        return avg_recall / n_query;
    }

    void save_recall(char *dataset, char *method_name, std::vector<std::pair<double, double>> time_recall_l,
                     std::vector<int> ef_search_l, int M,
                     int ef_construction, double indexing_time, long max_memory) {

        char base_path[256];
        std::sprintf(base_path, "../result/%s_%s_%d_%d", dataset, method_name, M, ef_construction);

        //mkdir
        if (mkdir(base_path, 0755) == -1) {
            printf("can not make directory, delete directory\n");
            char delete_command[256];
            std::sprintf(delete_command, "rm -rf %s", base_path);
            system(delete_command);
        }

        // write time-recall.txt
        std::ofstream outfile;
        char time_recall_path[256];
        std::sprintf(time_recall_path, "%s/time-recall.txt", base_path);
        outfile.open(time_recall_path);

        outfile << "time-per-query(ms), recall" << std::endl;
        for (int i = 0; i < time_recall_l.size(); i++) {
            outfile << time_recall_l[i].first << ", " << time_recall_l[i].second << std::endl;
        }
        outfile << std::endl;
        outfile.close();

        //write intermediate.json
        char intermediate_path[256];
        std::sprintf(intermediate_path, "%s/intermediate.json", base_path);
        outfile.open(intermediate_path);
        outfile << "{" << std::endl;
        outfile << "\"peak_memory\": " << max_memory << "," << std::endl;
        outfile << "\"indexing_build_time(ms)\": " << indexing_time << "," << std::endl;
        outfile << "\"M\": " << M << "," << std::endl;
        outfile << "\"ef_construction\": " << ef_construction << "," << std::endl;
        outfile << "\"ef_search_l\": [";
        for (int i = 0; i < ef_search_l.size(); i++) {
            outfile << ef_search_l[i];
            if (i != ef_search_l.size() - 1) {
                outfile << ", ";
            }
        }
        outfile << "]" << std::endl;

        outfile << "}" << std::endl;


    }

}
