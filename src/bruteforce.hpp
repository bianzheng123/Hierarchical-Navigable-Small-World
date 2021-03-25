#include <iostream>
#include "base.hpp"
#include "distance.hpp"
#include <vector>
#include <cstring>
#include <unordered_map>
#include <queue>

namespace MultipleHNSW {

    class BruteForceSearch : public AlgorithmInterface {
    public:
        BruteForceSearch(int vec_dim) : vec_dim_(vec_dim) {
        }

        void insert(const int *item, int label);

        int *search_by_label(int label);

        std::vector<int> search_knn(const int *query, int k);

        ~BruteForceSearch() {
            label2idx_.clear();
            idx2label_.clear();
            if (data_.size() != 0) {
                free(data_[0]);
            }
            data_.clear();
        }

    private:
        std::vector<int *> data_;
        //label和数组的idx是一一对应关系
        std::unordered_map<int, int> label2idx_;
        std::unordered_map<int, int> idx2label_;
        int vec_dim_;
    };

    void BruteForceSearch::insert(const int *item, int label) {
        if (label2idx_.find(label) != label2idx_.end()) { //find
            int item_idx = label2idx_[label];
            data_[item_idx] = (int *) item;
        } else { // not found in label2idx_
            label2idx_.insert({label, data_.size()});
            idx2label_.insert({data_.size(), label});
            data_.push_back((int *) item);
        }
    }

    int *BruteForceSearch::search_by_label(int label) {
        return data_[label2idx_[label]];
    }

    std::vector<int> BruteForceSearch::search_knn(const int *query, int k) {
        std::vector<int> result;
        int max_element = data_.size();

        std::priority_queue<distance_cmp, std::vector<distance_cmp>, distance_min_heap> queue;

        for (int i = 0; i < max_element; ++i) {
            long dis = l2distance(query, data_[i], vec_dim_);
            distance_cmp distance_item(dis, idx2label_[i], i);
            queue.push(distance_item);
        }
        for (int i = 0; i < k; i++) {
            distance_cmp p = queue.top();
            queue.pop();
            result.push_back(p.label_);
        }
        return result;
    }

}