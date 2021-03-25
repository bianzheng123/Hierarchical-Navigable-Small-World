import numpy as np
import faiss
import vecs_io


def get_neighbors(hnsw, i, level):
    " list the neighbors for node i at level "
    assert i < hnsw.levels.size()
    assert level < hnsw.levels.at(i)
    be = np.empty(2, 'uint64')
    hnsw.neighbor_range(i, level, faiss.swig_ptr(be), faiss.swig_ptr(be[1:]))
    return [hnsw.neighbors.at(j) for j in range(be[0], be[1])]


def get_graph(hnsw, level):
    graph = []
    for i in range(hnsw.levels.size()):
        if level >= hnsw.levels.at(i):
            tmp_neighbors = []
        else:
            tmp_neighbors = get_neighbors(hnsw, i, level)
            while -1 in tmp_neighbors:
                tmp_neighbors.remove(-1)
        graph.append(tmp_neighbors)

    return graph


M = 16
efConstruction = 100
efSearch = 100
random_seed = 100

print("faiss hnsw, M %d, efConstruction %d, efSearch %d" % (M, efConstruction, efSearch))

np.random.seed(random_seed)

base, d = vecs_io.bvecs_read('/home/bz/multiple-hnsw/siftsmall/base.bvecs')
base = base.astype(np.float32)
index = faiss.index_factory(int(base.shape[1]), "HNSW" + str(M))
hnsw = index.hnsw
hnsw.efConstruction = efConstruction
hnsw.efSearch = efSearch
index.add(base)
for i in range(index.hnsw.max_level):
    graph = get_graph(index.hnsw, i)
    # print(graph)
    max_m = -1
    max_m_idx = -1
    total_edge = 0
    for j, edge in enumerate(graph, 0):
        total_edge += len(edge)
        if len(edge) > max_m:
            max_m = len(edge)
            max_m_idx = j
    # print(graph[0])
    print("at level %d, n_edges %d largest M %d idx %d" % (i, total_edge, max_m, max_m_idx))
    if i == 0:
        for edge in graph:
            if len(edge) <= 0:
                print("the edge should be greater than 0 at bottom level")
