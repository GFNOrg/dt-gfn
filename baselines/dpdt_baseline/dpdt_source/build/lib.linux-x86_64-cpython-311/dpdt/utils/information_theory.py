import numpy as np
from dpdt.utils.datasets import Data
from scipy.stats import entropy
from sklearn.feature_selection import f_classif


def info_gain(data: Data, split_feature: int, split_value: float, entr_parent: float):
    assert isinstance(split_value, float), print(data.x.shape)
    # Gain = (Entropy of the parent node) – (average entropy of the child nodes)
    test = data.x[:, split_feature] <= split_value
    inf = np.nonzero(test)
    sup = np.nonzero(test - 1)
    data_left = Data(data.x[inf], data.y[inf])
    # print(data_left.x.shape)
    data_right = Data(data.x[sup], data.y[sup])

    _, counts_left = np.unique(data_left.y, return_counts=True)
    _, counts_right = np.unique(data_right.y, return_counts=True)

    entropy_left = entropy(counts_left / data_left.y.shape[0])
    entropy_right = entropy(counts_right / data_right.y.shape[0])

    p_left = data_left.x.shape[0] / data.x.shape[0]
    p_right = 1 - p_left
    return (
        entr_parent - (p_left * entropy_left + p_right * entropy_right),
        data_left,
        data_right,
        p_left,
        p_right,
    )


def get_info_gains(
    data: Data,
    k: int,
    nb_feats: int = None,
    nb_samples: int = None,
    feature_select: str = "fclassif",
):
    igs = []
    feat_thresh = []
    lefts, rights = [], []
    pls, prs = [], []
    # regroup dataset features
    idx = 0
    _, counts = np.unique(data.y, return_counts=True)
    entropy_parent = entropy(counts / data.y.shape[0])

    if feature_select == "fclassif":
        assert nb_samples is not None and nb_feats is not None

        idx_samples = np.random.choice(
            np.arange(data.x.shape[0]),
            size=min(nb_samples, data.x.shape[0]),
            replace=False,
        )
        x_new, y_new = data.x[idx_samples].copy(), data.y[idx_samples].copy()
        scores_ = f_classif(x_new, y_new)
        feats = np.argsort(scores_)[0][max(-nb_feats, -data.x.shape[1]) :]
    elif feature_select == "rnd":
        assert nb_samples is not None and nb_feats is not None

        # select nb_features at random
        feats = np.random.choice(
            np.arange(data.x.shape[1]),
            size=min(nb_feats, data.x.shape[1]),
            replace=False,
        )
        nb_samples = min(nb_samples, data.x.shape[0])
        x_new = np.zeros((nb_samples, data.x.shape[1]))
        for feat in feats:
            x_new[:, feat] = np.random.choice(
                data.x[:, feat], size=nb_samples, replace=False
            )

    elif feature_select == "":
        x_new = data.x
        feats = np.arange(data.x.shape[1], dtype=np.uint8)
    for feat in feats:
        for threshold in x_new[:, feat]:
            ig, left, right, pl, pr = info_gain(data, feat, threshold, entropy_parent)
            igs.append(ig)
            feat_thresh.append([feat, threshold])
            lefts.append(left)
            rights.append(right)
            pls.append(pl)
            prs.append(pr)
            idx += 1

    sorted_feat_thresh = []
    sorted_lefts, sorted_rights = [], []
    sorted_plefts, sorted_prights = [], []
    # sort feat_tresh
    for idx in np.argsort(np.array(igs))[max(-k, -len(igs)) :]:  # small to big
        sorted_feat_thresh.append(feat_thresh[idx])
        sorted_lefts.append(lefts[idx])
        sorted_rights.append(rights[idx])
        sorted_plefts.append(pls[idx])
        sorted_prights.append(prs[idx])
    return (
        sorted_feat_thresh,
        sorted_lefts,
        sorted_rights,
        sorted_plefts,
        sorted_prights,
    )


def get_info_gains_count_based(
    data: Data,
    k: int,
    nb_samples: int = None,
):
    igs = []
    feat_thresh = []
    # regroup dataset features
    if nb_samples is not None:
        idx_samples = np.random.choice(
            np.arange(data.x.shape[0]),
            size=min(nb_samples, data.x.shape[0]),
            replace=False,
        )
        datax, datay = data.x[idx_samples], data.y[idx_samples]
    else:
        datax, datay = data.x, data.y

    classes, counts_temp = np.unique(datay, return_counts=True)
    counts = np.zeros(max(classes) + 1)
    for cl, co in zip(classes, counts_temp):
        counts[cl] = co
    entropy_parent = entropy(counts)
    dsize = datax.shape[0]
    for feat in range(datax.shape[1]):
        # sort according to feature 'feat'
        idxs = np.argsort(
            datax[:, feat]
        )  # ignore last data point (not useful data split)
        sorted_x = datax[idxs[:-1], feat]
        sorted_y = datay[idxs[:-1]]
        counts_right = counts.copy()  # initially, all data on right child
        counts_left = np.zeros_like(counts)
        for id, (threshold, ythresh) in enumerate(zip(sorted_x, sorted_y)):
            counts_left[ythresh] += 1
            counts_right[ythresh] -= 1
            suml, sumr = sum(counts_left), sum(counts_right)
            pl = suml / dsize
            pr = 1.0 - pl
            ig = entropy_parent - (
                pl * entropy(counts_left) + pr * entropy(counts_right)
            )
            igs.append(ig)
            feat_thresh.append([feat, threshold])

    igidxs = np.argsort(igs)[max(-k, -len(igs)) :]
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
        prs.append(1.0 - pl)

    return best_splits, lefts, rights, pls, prs
