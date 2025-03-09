import numpy as np

from .datasets import Data

cimport numpy as cnp

cnp.import_array()
DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t
ctypedef cnp.float64_t DTYPEDOUBLE_t
ctypedef cnp.uint8_t DTYPEUINT8_t
cimport cython

from cython.cimports.libc.math import log2


# @cython.cdivision(True)
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# @cython.exceptval(check=False)
# cdef float entropy(cnp.ndarray[DTYPE_t, ndim=1] c, int n, int t):
#     cdef float e, v, p
#     e = 0
#     cdef int i
#     for i in range(n):
#         v = c[i]
#         if v != 0:
#             p = v / t
#             e += p * log2(p)
#     return -e

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef float entropy(cnp.ndarray[DTYPE_t, ndim=1] c, int n, int t):
    cdef float e, v, p
    e = 0
    cdef int i
    for i in range(n):
        v = c[i]
        if v != 0:
            p = v / t
            e += p * log2(p)
    return -e

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cy_get_info_gains_count_based(
        data: Data,
        k: int,
        nb_samples: int = None,
):
    igs = []
    feat_thresh = []
    lefts, rights = [], []
    pls, prs = [], []
    cdef cnp.ndarray[DTYPEDOUBLE_t, ndim=2] datax
    cdef cnp.ndarray[DTYPEUINT8_t, ndim=1] datay
    # regroup dataset features
    if nb_samples is not None:
        idx_samples = np.random.choice(np.arange(data.x.shape[0]), size=min(nb_samples, data.x.shape[0]), replace=False)
        datax, datay = data.x[idx_samples], data.y[idx_samples]
    else:
        datax, datay = data.x, data.y

    classes, counts_temp = np.unique(datay, return_counts=True)
    cdef int nb_class = max(classes) + 1
    cdef cnp.ndarray[DTYPE_t, ndim=1] counts = np.zeros(nb_class, dtype=DTYPE)
    cdef int cl, co
    for cl, co in zip(classes, counts_temp):
        counts[cl] = co
    cdef int dsize = datax.shape[0]
    cdef int suml, sumr
    cdef cnp.ndarray[DTYPE_t, ndim=1] counts_right
    cdef cnp.ndarray[DTYPE_t, ndim=1] counts_left
    cdef float pl, pr
    entropy_parent = entropy(counts, nb_class, dsize)
    for feat in range(datax.shape[1]):
        # sort according to feature 'feat'
        suml = 0
        sumr = dsize
        idxs = np.argsort(datax[:, feat])  # ignore last data point (not useful data split)
        sorted_x = datax[idxs[:-1], feat]
        sorted_y = datay[idxs[:-1]]
        counts_right = counts.copy()  # initially, all data on right child
        counts_left = np.zeros_like(counts)
        for id, (threshold, ythresh) in enumerate(zip(sorted_x, sorted_y)):
            counts_left[ythresh] += 1
            counts_right[ythresh] -= 1
            suml += 1
            sumr -= 1
            pl = suml / dsize
            pr = 1. - pl
            ig = entropy_parent - (pl * entropy(counts_left, nb_class, suml) + pr * entropy(counts_right, nb_class, sumr))
            igs.append(ig)
            feat_thresh.append([feat, threshold])

    igidxs = np.argsort(igs)[len(igs) + max(-k, -len(igs)):]
    best_splits, lefts, rights, pls, prs = [], [], [], [], []

    for i in igidxs:
        best_splits.append(feat_thresh[i])
        feat_idx, feat_val = feat_thresh[i]
        data_filter = data.x[:, feat_idx] <= feat_val
        neg_data_filter = ~data_filter
        lefts.append(Data(data.x[data_filter], data.y[data_filter]))
        rights.append(Data(data.x[neg_data_filter], data.y[neg_data_filter]))
        pl = sum(data_filter) / data.x.shape[0]
        pls.append(pl)
        prs.append(1. - pl)

    return best_splits, lefts, rights, pls, prs


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cy_last_depth_count_based(
        cnp.ndarray[DTYPEDOUBLE_t, ndim=2] datasx,
        cnp.ndarray[DTYPEUINT8_t, ndim=1] datasy,
        filter,
):
    cdef cnp.ndarray[DTYPEDOUBLE_t, ndim = 2] datax = datasx[filter]
    cdef cnp.ndarray[DTYPEUINT8_t, ndim = 1] datay = datasy[filter]
    cdef int best_feat = 0
    cdef double best_thresh = 0.
    cdef DTYPE_t dsize = datax.shape[0]
    cdef DTYPE_t best_loss = dsize
    cdef DTYPE_t max_left, max_right, lj, rj = 0
    cdef int nbfeat = datax.shape[1]
    cdef DTYPEDOUBLE_t threshold
    cdef DTYPEUINT8_t ythresh

    # counting class occurrences in data
    classes, counts_temp = np.unique(datay, return_counts=True)
    cdef int nb_class = max(classes) + 1
    cdef cnp.ndarray[DTYPE_t, ndim=1] counts = np.zeros(nb_class, dtype=DTYPE)
    cdef int cl, co
    for cl, co in zip(classes, counts_temp):
        counts[cl] = co

    # counts for left and right childs after split
    cdef cnp.ndarray[DTYPE_t, ndim=1] counts_right
    cdef cnp.ndarray[DTYPE_t, ndim=1] counts_left
    cdef int k = 0
    cdef int j = 0
    cdef cnp.ndarray[DTYPEDOUBLE_t, ndim = 1] sorted_x
    cdef cnp.ndarray[DTYPEUINT8_t, ndim = 1] sorted_y
    for feat in range(nbfeat):
        # sort according to feature 'feat'
        idxs = np.argsort(datax[:, feat])  # ignore last data point (not useful data split)
        sorted_x = datax[idxs[:-1], feat]
        sorted_y = datay[idxs[:-1]]
        counts_right = counts.copy()  # initially, all data on right child
        counts_left = np.zeros_like(counts)
        for k in range(dsize - 1):
            threshold, ythresh = sorted_x[k], sorted_y[k]
            counts_left[ythresh] += 1
            counts_right[ythresh] -= 1
            max_left = 0
            max_right = 0
            for j in range(nb_class):
                lj, rj = counts_left[j], counts_right[j]
                if lj > max_left:
                    max_left = lj
                if rj > max_right:
                    max_right = rj
            loss = dsize - max_left - max_right
            if loss < best_loss:
                best_loss = loss
                best_feat = feat
                best_thresh = threshold

    best_splits = [[best_feat, best_thresh]]
    split_left = datasx[:, best_feat] <= best_thresh
    lefts = [split_left * filter]
    rights = [~split_left * filter]
    pls = [sum(lefts[0]) / dsize]
    prs = [1. - pls[0]]
    return best_splits, lefts, rights, pls, prs


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cy_get_info_gains_quantile_count_based(
        cnp.ndarray[DTYPEDOUBLE_t, ndim=2] datasx,
        cnp.ndarray[DTYPEUINT8_t, ndim=1] datasy,
        filter,
        int nb_iga,
        int quantile,
):
    cdef cnp.ndarray[DTYPEDOUBLE_t, ndim = 2] datax = datasx[filter]
    cdef cnp.ndarray[DTYPEUINT8_t, ndim = 1] datay = datasy[filter]
    igs = []
    feat_thresh = []
    lefts, rights = [], []
    pls, prs = [], []

    classes, counts_temp = np.unique(datay, return_counts=True)
    cdef int nb_class = max(classes) + 1
    cdef cnp.ndarray[DTYPE_t, ndim=1] counts = np.zeros(nb_class, dtype=DTYPE)
    cdef int cl, co
    for cl, co in zip(classes, counts_temp):
        counts[cl] = co
    cdef DTYPE_t dsize = datax.shape[0]
    cdef int pid
    cdef int nbfeat = datax.shape[1]
    cdef float inter_size = max(dsize / quantile, 1.)
    cdef DTYPE_t last_idx_test = min(int(inter_size * (quantile - 1)), dsize - 1)
    cdef DTYPE_t next_idx_test, nb_idx_test
    cdef DTYPE_t suml, sumr
    cdef cnp.ndarray[DTYPE_t, ndim=1] counts_right
    cdef cnp.ndarray[DTYPE_t, ndim=1] counts_left
    cdef float pl, pr
    cdef double threshold
    cdef DTYPEUINT8_t ythresh
    cdef cnp.ndarray[DTYPEDOUBLE_t, ndim = 1] sorted_x
    cdef cnp.ndarray[DTYPEUINT8_t, ndim = 1] sorted_y
    for feat in range(nbfeat):
        # sort according to feature 'feat'
        idxs = np.argsort(datax[:, feat])
        sorted_x = datax[idxs, feat]
        sorted_y = datay[idxs]
        counts_right = counts.copy()  # initially, all data on right child
        counts_left = np.zeros_like(counts)
        next_idx_test = int(inter_size)
        nb_idx_test = 1
        for pid in range(dsize):
            ythresh = sorted_y[pid]
            counts_left[ythresh] += 1
            counts_right[ythresh] -= 1
            if (pid + 1) == next_idx_test:
                nb_idx_test += 1
                next_idx_test = int(inter_size * nb_idx_test)
                threshold = sorted_x[pid]
                suml = pid + 1
                sumr = dsize - suml
                pl = suml / dsize
                pr = 1. - pl
                ig = -(pl * entropy(counts_left, nb_class, suml) + pr * entropy(counts_right, nb_class, sumr))
                igs.append(ig)
                feat_thresh.append([feat, threshold])
                if (pid + 1) == last_idx_test:
                    break

    igidxs = np.argsort(igs)[len(igs) + max(-nb_iga, -len(igs)):]
    best_splits, lefts, rights, pls, prs = [], [], [], [], []
    for i in igidxs:
        best_splits.append(feat_thresh[i])
        feat_idx, feat_val = feat_thresh[i]

        split_left = datasx[:, feat_idx] <= feat_val
        left = split_left * filter
        lefts.append(left)
        rights.append(~split_left * filter)

        pl = sum(left) / dsize
        pls.append(pl)
        prs.append(1. - pl)
    return best_splits, lefts, rights, pls, prs