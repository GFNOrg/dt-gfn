from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import entropy
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# from dpdt.utils.cy_feature_select import cy_get_info_gains_quantile_count_based


class AIGSelector(ABC):
    @abstractmethod
    def select(self):
        pass


class KBest(AIGSelector):
    def __init__(self, k: int = 1):
        self.k = k

    def select(self, nz):
        igs = []
        feat_thresh = []
        lefts, rights = [], []
        pls, prs = [], []
        _, counts = np.unique(self.data.y[nz], return_counts=True)
        entropy_parent = entropy(counts / self.data.y[nz].shape[0])
        for feat in range(self.data.x[nz].shape[1]):
            for threshold in self.data.x[nz][:, feat]:
                ig, left, right, pl, pr = self.info_gain(
                    feat, threshold, entropy_parent, nz
                )
                igs.append(ig)
                feat_thresh.append([feat, threshold])
                lefts.append(left)
                rights.append(right)
                pls.append(pl)
                prs.append(pr)

        sorted_feat_thresh = []
        sorted_lefts, sorted_rights = [], []
        sorted_plefts, sorted_prights = [], []
        # sort feat_tresh
        for idx in np.argsort(np.array(igs))[max(-self.k, -len(igs)) :]:  # small to big
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

    def info_gain(self, split_feature, split_value, entr_parent, nz):

        inf = (self.data.x[:, split_feature] <= split_value) * nz
        sup = np.logical_not(inf) * nz
        p_left = inf.sum() / nz.sum()
        p_right = 1 - p_left

        # print(data_left.x.shape)

        _, counts_left = np.unique(self.data.y[inf], return_counts=True)
        _, counts_right = np.unique(self.data.y[sup], return_counts=True)

        entropy_left = entropy(counts_left / self.data.y[inf].shape[0])
        entropy_right = entropy(counts_right / self.data.y[sup].shape[0])

        return (
            entr_parent - (p_left * entropy_left + p_right * entropy_right),
            inf,
            sup,
            p_left,
            p_right,
        )


class CartAIGSelector(AIGSelector):
    def __init__(self, depth: int = 4, max_tree_sizes: list = None):
        self.depth = depth
        self.max_tree_sizes = max_tree_sizes
        self.counter_depths = 0

    def select(self, nz):
        feat_thresh = []
        lefts, rights = [], []
        pls, prs = [], []
        if self.max_tree_sizes is not None:
            clf = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=self.max_tree_sizes[self.counter_depths],
                random_state=0,
            )
        else:
            clf = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=self.depth,
                random_state=0,
            )
        self.counter_depths += 1

        clf.fit(self.data.x[nz], self.data.y[nz])
        for i in range(len(clf.tree_.feature)):
            if clf.tree_.feature[i] >= 0:
                inf = (
                    self.data.x[:, clf.tree_.feature[i]] <= clf.tree_.threshold[i]
                ) * nz
                sup = np.logical_not(inf) * nz
                p_left = inf.sum() / nz.sum()
                p_right = 1 - p_left
                lefts.append(inf)
                rights.append(sup)
                feat_thresh.append([clf.tree_.feature[i], clf.tree_.threshold[i]])
                pls.append(p_left)
                prs.append(p_right)
        return feat_thresh, lefts, rights, pls, prs


class CartAIGSelectorReg(AIGSelector):
    def __init__(self, depth: int = 4, max_tree_sizes: list = None):
        self.depth = depth
        self.max_tree_sizes = max_tree_sizes
        self.counter_depths = 0

    def select(self, nz):
        feat_thresh = []
        lefts, rights = [], []
        pls, prs = [], []
        if self.max_tree_sizes is not None:
            clf = DecisionTreeRegressor(
                max_depth=self.max_tree_sizes[self.counter_depths],
                random_state=0,
            )
        else:
            clf = DecisionTreeRegressor(
                max_depth=self.depth,
                random_state=0,
            )
        self.counter_depths += 1

        clf.fit(self.data.x[nz], self.data.y[nz])
        for i in range(len(clf.tree_.feature)):
            if clf.tree_.feature[i] >= 0:
                inf = (
                    self.data.x[:, clf.tree_.feature[i]] <= clf.tree_.threshold[i]
                ) * nz
                sup = np.logical_not(inf) * nz
                p_left = inf.sum() / nz.sum()
                p_right = 1 - p_left
                lefts.append(inf)
                rights.append(sup)
                feat_thresh.append([clf.tree_.feature[i], clf.tree_.threshold[i]])
                pls.append(p_left)
                prs.append(p_right)
        return feat_thresh, lefts, rights, pls, prs


class DeepTreeCartAIGSelector(AIGSelector):
    def __init__(self, max_depth_dp: int):
        self.max_depth = max_depth_dp

    def select(self, nz, current_dp_depth):
        feat_thresh = []
        lefts, rights = [], []
        pls, prs = [], []

        clf = DecisionTreeClassifier(
            criterion="entropy", max_depth=self.max_depth - current_dp_depth
        )

        clf.fit(self.data.x[nz], self.data.y[nz])
        for i in range(len(clf.tree_.feature)):
            inf = self.data.x[:, clf.tree_.feature[i]] <= clf.tree_.threshold[i] * nz
            sup = np.logical_not(inf) * nz
            p_left = inf.sum() / nz.sum()
            p_right = 1 - p_left
            lefts.append(inf)
            rights.append(sup)
            feat_thresh.append([clf.tree_.feature[i], clf.tree_.threshold[i]])
            pls.append(p_left)
            prs.append(p_right)
        return feat_thresh, lefts, rights, pls, prs


# class QuantileInfoGainAIGSelector(AIGSelector):
#     def __init__(self, nb_aig, quantile):
#         self.nb_aig = nb_aig
#         self.quantile = quantile

#     def select(self, nz):
#         return cy_get_info_gains_quantile_count_based(self.data.x, self.data.y, nz, self.nb_aig, self.quantile)
