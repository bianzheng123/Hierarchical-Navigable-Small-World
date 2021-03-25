#include <iostream>
#include "base.hpp"
#include "distance.hpp"
#include <vector>
#include <cstring>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <cassert>
#include <algorithm>

namespace MultipleHNSW {

    class HNSW : public AlgorithmInterface {
    public:
        HNSW(int vec_dim, int max_element, int M = 16, int ef_construction = 200, int ef_search = 200,
             int random_seed = 100) {
            max_element_ = max_element;
            vec_dim_ = vec_dim;
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = ef_construction;
            ef_search_ = ef_search;
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
            maxlevel_ = -1;
            enterpoint_node_ = -1;

            level_generator_.seed(random_seed);
        }

        void insert(const int *item, int label);

        int *search_by_label(int label);

        std::vector<int> search_knn(const int *query, int k);

        void test_graph();

        ~HNSW() {
            label2idx_.clear();
            idx2label_.clear();
            if (data_.size() != 0) {
                free(data_[0]);
            }
            data_.clear();
            idx_levels.clear();
        }

        int M_;
        int ef_construction_;
        int ef_search_;

        int test_n_cand_ = 0;
        int test_n_neighbor_ = 0;

    private:
        std::vector<int *> data_;
        //label和数组的idx是一一对应关系
        std::unordered_map<int, int> label2idx_;
        std::unordered_map<int, int> idx2label_;
        //索引是元素的idx, 值是其对应的层数
        std::vector<int> idx_levels;
        std::vector<std::vector<std::vector<int>>> hierarchical_graph_;
        //第一个vector代表第几层, 第二个vector代表第几个item, 第三个vector代表该item下的邻居节点
        int max_element_;
        int vec_dim_;
        int maxM_, maxM0_;
        double mult_, revSize_;
        int maxlevel_;
        int enterpoint_node_;

        std::default_random_engine level_generator_;

        int get_random_level();

        std::vector<int>
        search_layer(const int *item, std::vector<int> enterpoint_l, int ef_construction, int layer);

        std::vector<int> select_neighbors(int item_idx, std::vector<int> cand, int M, int layer);

    };

    int HNSW::get_random_level() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * mult_;
        return (int) r;
    }

    void HNSW::insert(const int *item, int label) {
        if (label2idx_.find(label) != label2idx_.end()) { //the label exists, directly replace
            int item_idx = label2idx_[label];
            data_[item_idx] = (int *) item;
            return;
        }
        int item_idx = data_.size();
        //label not found in label2idx_
        label2idx_.insert({label, item_idx});
        idx2label_.insert({item_idx, label});
        data_.push_back((int *) item);

        int curlevel = get_random_level();
        idx_levels.push_back(curlevel);

        int currObj = enterpoint_node_;
        if (currObj != -1) {
            std::vector<int> ep_l;
            ep_l.push_back(currObj);
            for (int i = maxlevel_; i >= curlevel + 1; --i) {
                ep_l = search_layer(item, ep_l, 1, i);
            }
            int min_val = curlevel < maxlevel_ ? curlevel : maxlevel_;
            for (int i = min_val; i >= 0; --i) {
                std::vector<int> cand = search_layer(item, ep_l, ef_construction_, i);
                test_n_cand_ = cand.size() > test_n_cand_ ? cand.size() : test_n_cand_;
                std::vector<int> neighbors = select_neighbors(item_idx, cand, M_, i);
                test_n_neighbor_ = neighbors.size() > test_n_neighbor_ ? neighbors.size() : test_n_neighbor_;
                for (int neighbor: neighbors) {
                    hierarchical_graph_[i][neighbor].push_back(item_idx);
                    hierarchical_graph_[i][item_idx].push_back(neighbor);
                }
                for (int e: neighbors) {
                    std::vector<int> neighborhood = hierarchical_graph_[i][e];
                    int currMaxM = i == 0 ? maxM0_ : maxM_;
                    if (neighborhood.size() > currMaxM) {
                        std::vector<int> eNewConn = select_neighbors(e, hierarchical_graph_[i][e], currMaxM, i);
                        hierarchical_graph_[i][e] = eNewConn;
                    }
                }
                ep_l = cand;
            }
        } else {
            //init enterpoint
            enterpoint_node_ = item_idx;
            maxlevel_ = curlevel;
            hierarchical_graph_ = std::vector<std::vector<std::vector<int>>>(curlevel + 1);
            for (int i = 0; i < curlevel + 1; i++) {
                hierarchical_graph_[i].resize(max_element_);
            }
        }

        if (curlevel > maxlevel_) {
            for (int i = 0; i < curlevel - maxlevel_; i++) {
                std::vector<std::vector<int>> tmp_graph(max_element_);
                hierarchical_graph_.push_back(tmp_graph);
            }
            enterpoint_node_ = item_idx;
            maxlevel_ = curlevel;
        }

    }

    std::vector<int> HNSW::select_neighbors(int item_idx, std::vector<int> cand, int M, int layer) {
        std::priority_queue<distance_cmp, std::vector<distance_cmp>, distance_min_heap> working_queue;
        std::vector<int> r_queue;
        std::priority_queue<distance_cmp, std::vector<distance_cmp>, distance_min_heap> working_discard_queue;
        for (int i = 0; i < cand.size(); i++) {
            long tmp_dist = l2distance(data_[item_idx], data_[cand[i]], vec_dim_);
            distance_cmp tmp(tmp_dist, idx2label_[cand[i]], cand[i]);
            working_queue.push(tmp);
        }
        while (working_queue.size() > 0 and r_queue.size() < M) {
            distance_cmp neighbor_tmp = working_queue.top();
            working_queue.pop();
            if (r_queue.size() == 0) {
                r_queue.push_back(neighbor_tmp.idx_);
            }
            bool closerToAnyElement = true;
            for (int i = 0; i < r_queue.size(); ++i) {
                if (neighbor_tmp.distance_ > l2distance(data_[neighbor_tmp.idx_], data_[r_queue[i]], vec_dim_)) {
                    closerToAnyElement = false;
                    break;
                }
            }
            if (closerToAnyElement) {
                r_queue.push_back(neighbor_tmp.idx_);
            } else {
                working_discard_queue.push(neighbor_tmp);
            }
        }

        //add some discarded, keepPrunedConnections
        if (false) {
            while (working_discard_queue.size() > 0 && r_queue.size() < M) {
                distance_cmp neighbor_tmp = working_discard_queue.top();
                working_discard_queue.pop();
                r_queue.push_back(neighbor_tmp.idx_);
            }
        }

        return r_queue;
    }

    //返回的数组是按照相对query距离排序的, 降序, 就是第一个元素是数组中离query最远的元素
    std::vector<int> HNSW::search_layer(const int *item, std::vector<int> enterpoint_l, int ef_construction, int layer) {
        std::unordered_set<int> visited_element;
        std::priority_queue<distance_cmp, std::vector<distance_cmp>, distance_min_heap> candidates;
        std::priority_queue<distance_cmp, std::vector<distance_cmp>, distance_max_heap> nearest_neighbors;

        for (int i = 0; i < enterpoint_l.size(); i++) {
            visited_element.insert(enterpoint_l[i]);
            long tmp_dist = l2distance(item, data_[enterpoint_l[i]], vec_dim_);
            distance_cmp tmp = distance_cmp(tmp_dist, idx2label_[enterpoint_l[i]], enterpoint_l[i]);
            candidates.push(tmp);
            nearest_neighbors.push(tmp);
        }

        while (candidates.size() > 0) {
            distance_cmp cand = candidates.top();
            candidates.pop();
            distance_cmp far_ele = nearest_neighbors.top();
            if (l2distance(item, data_[cand.idx_], vec_dim_) >
                l2distance(item, data_[far_ele.idx_], vec_dim_)) {
                break;
            }
            for (int neighbor: hierarchical_graph_[layer][cand.idx_]) {
                if (visited_element.find(neighbor) == visited_element.end()) { //can not find
                    visited_element.insert(neighbor);
                    far_ele = nearest_neighbors.top();
                    if (l2distance(item, data_[neighbor], vec_dim_) <
                        l2distance(item, data_[far_ele.idx_], vec_dim_) ||
                        nearest_neighbors.size() < ef_construction) {
                        long tmp_dist = l2distance(item, data_[neighbor], vec_dim_);
                        distance_cmp tmp = distance_cmp(tmp_dist, idx2label_[neighbor], neighbor);
                        candidates.push(tmp);
                        nearest_neighbors.push(tmp);
                        while (nearest_neighbors.size() > ef_construction) {
                            nearest_neighbors.pop();
                        }
                    }
                }
            }
        }
        while (nearest_neighbors.size() > ef_construction) {
            nearest_neighbors.pop();
        }

        std::vector<int> res;
        while (nearest_neighbors.size() > 0) {
            distance_cmp tmp = nearest_neighbors.top();
            nearest_neighbors.pop();
            res.push_back(tmp.idx_);
        }

        if (res.size() > ef_construction) {
            printf("%d %d %d\n", res.size(), ef_construction, layer);
//            printf("error, in function search layer, the size of result is not equal to ef_construction");
        }
        return res;
    }

    void HNSW::test_graph() {
        assert(hierarchical_graph_.size() != maxlevel_);
        assert(hierarchical_graph_[0].size() != max_element_);
        for (int i = 0; i < maxlevel_; i++) {
            int n_edges = 0;
            int biggestM = 0;
            int biggestMidx = -1;
            for (int j = 0; j < max_element_; j++) {
                n_edges += hierarchical_graph_[i][j].size();
                if (hierarchical_graph_[i][j].size() > biggestM) {
                    biggestM = hierarchical_graph_[i][j].size();
                    biggestMidx = j;
                }
            }
            printf("In level %d, n_edges %d, biggestM %d, biggestMidx %d\n", i, n_edges, biggestM, biggestMidx);
        }
        for (int i = 0; i < max_element_; i++) {
            assert(hierarchical_graph_[0][i].size() > 0);
        }
        printf("During insertion, max cand %d max n_neighbor %d\n", test_n_cand_, test_n_neighbor_);
    }

    int *HNSW::search_by_label(int label) {
        return data_[label2idx_[label]];
    }

    std::vector<int> HNSW::search_knn(const int *query, int k) {
        std::vector<int> result;
        int max_element = data_.size();

        std::vector<int> nearest_ele;

        int currObj = enterpoint_node_;
        int enterPointLevel = idx_levels[currObj];
        for (int currLevel = enterPointLevel; currLevel > 0; currLevel--) {
            nearest_ele.push_back(currObj);
            nearest_ele = search_layer(query, nearest_ele, 1, currLevel);
        }
        nearest_ele = search_layer(query, nearest_ele, ef_search_, 0);

        std::vector<int> res;
        for (int i = 0; i < k; i++) {
            res.push_back(nearest_ele[nearest_ele.size() - i - 1]);
        }
        return res;
    }

}