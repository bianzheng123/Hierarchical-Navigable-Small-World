
namespace MultipleHNSW {

    class distance_cmp{
    public:
        long distance_;
        int label_;
        int idx_;
        distance_cmp(int distance, int label, int idx): distance_(distance), label_(label), idx_(idx){}
        distance_cmp(){
            distance_ = 0;
            label_ = 0;
            idx_ = -1;
        }
    };

    struct distance_min_heap {
        bool operator()(distance_cmp a, distance_cmp b) {
            return a.distance_ > b.distance_; //<大顶堆, >小顶堆
        }
    };

    struct distance_max_heap {
        bool operator()(distance_cmp a, distance_cmp b) {
            return a.distance_ < b.distance_; //<大顶堆, >小顶堆
        }
    };

    long l2distance(const int *a, const int *b, int vec_dim) {
        long dis = 0;
        for (int i = 0; i < vec_dim; ++i) {
            long tmp = a[i] - b[i];
            dis += (tmp * tmp);
        }
        return dis;
    }
}