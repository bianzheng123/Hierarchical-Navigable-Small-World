import numpy as np
import struct


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy(), d


def fvecs_read(fname):
    data, d = ivecs_read(fname)
    return data.view('float32').astype(np.float32), d


def bvecs_read(fname):
    a = np.fromfile(fname, dtype='uint8')
    d = a[:4].view('uint8')[0]
    return a.reshape(-1, d + 4)[:, 4:].copy(), d


# put the part of file into cache, prevent the slow load that file is too big
def fvecs_read_mmap(fname):
    x = np.memmap(fname, dtype='int32', mode='r', order='C')
    # x = np.memmap(fname, dtype='int32')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:], d


def bvecs_read_mmap(fname):
    x = np.memmap(fname, dtype='uint8', mode='r', order='C')
    # x = np.memmap(fname, dtype='uint8')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:], d


def ivecs_read_mmap(fname):
    x = np.memmap(fname, dtype='int32', mode='r', order='C')
    # x = np.memmap(fname, dtype='int32')
    d = x[0]
    return x.reshape(-1, d + 1)[:, 1:], d


# store in format of vecs
def fvecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))  # *dimension就是int, dimension就是list
        f.write(struct.pack('f' * len(x), *x))

    f.close()


def ivecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('i' * len(x), *x))

    f.close()


def bvecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('B' * len(x), *x))

    f.close()


query, d = fvecs_read('/home/bz/Dataset/sift/sift_query.fvecs')
# np.savetxt('query.txt', query, fmt='%.3f')

base, d = fvecs_read('/home/bz/Dataset/sift/sift_base.fvecs')
# np.savetxt('base.txt', base, fmt='%.3f')

gnd, d = ivecs_read('/home/bz/Dataset/sift/sift_groundtruth.ivecs')
# np.savetxt('gnd.txt', gnd, fmt='%.3f')

query = query[:1000].astype(np.int)
base = base.astype(np.int)
gnd = gnd[:,:10].astype(np.int)
bvecs_write('/home/bz/multiple-hnsw/sift/query.bvecs', query)
bvecs_write('/home/bz/multiple-hnsw/sift/base.bvecs', base)
ivecs_write('/home/bz/multiple-hnsw/sift/gnd.ivecs', gnd)

