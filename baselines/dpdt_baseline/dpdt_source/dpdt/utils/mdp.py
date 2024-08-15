from typing import Union

import sys
import os

# Add the dpdt_source directory to the Python path
current_dir = os.path.dirname(__file__)
dpdt_source_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(dpdt_source_path)

import numpy as np
from sklearn.metrics import zero_one_loss, mean_squared_error

from dpdt.utils.datasets import Data
from dpdt.utils.feature_selectors import AIGSelector
from dpdt.utils.cy_feature_select import cy_last_depth_count_based as last_depth_select


def eval_in_mdp(
    S: np.ndarray, Y: np.ndarray, policy: dict, init_o: np.ndarray, zeta: int
):
    nb_features = S.shape[1]
    score = 0
    visited_nodes = set()
    init_a = policy[tuple(init_o.tolist() + [0])][zeta]
    for i, s in enumerate(S):
        a = init_a
        o = init_o.copy()
        H = 0
        while not isinstance(a, np.uint8):  # a is int implies leaf node
            visited_nodes.add(tuple(o.tolist() + [H]))
            feature, threshold = a
            H += 1
            if s[feature] <= threshold:
                o[nb_features + feature] = threshold
            else:
                o[feature] = threshold
            a = policy[tuple(o.tolist() + [H])][zeta]
        visited_nodes.add(tuple(o.tolist() + [H]))  # Add leaf node
        score += a == Y[i]
    
    total_nodes = len(visited_nodes)
    return score / S.shape[0], total_nodes


def average_traj_length_in_mdp(
    S: np.ndarray, Y: np.ndarray, policy: dict, init_o: np.ndarray, zeta: int
):
    nb_features = S.shape[1]
    score = 0
    init_a = policy[tuple(init_o.tolist() + [0])][zeta]
    lengths = np.zeros(S.shape[0])
    for i, s in enumerate(S):
        a = init_a
        o = init_o.copy()
        H = 0
        while not isinstance(a, np.uint8):  # a is int implies leaf node
            feature, threshold = a
            H += 1
            if s[feature] <= threshold:
                o[nb_features + feature] = threshold
            else:
                o[feature] = threshold
            a = policy[tuple(o.tolist() + [H])][zeta]
            # print(s, o, a)
        score += a == Y[i]
        lengths[i] = H
    return score / S.shape[0], lengths.mean()

def average_traj_length_in_mdp_regression(
    S: np.ndarray, Y: np.ndarray, policy: dict, init_o: np.ndarray, zeta: int
):
    nb_features = S.shape[1]
    score = 0
    init_a = policy[tuple(init_o.tolist() + [0])][zeta]
    lengths = np.zeros(S.shape[0])
    preds = []
    for i, s in enumerate(S):
        a = init_a
        o = init_o.copy()
        H = 0
        while not isinstance(a, np.ndarray):  # a is int implies leaf node
            feature, threshold = a
            H += 1
            if s[feature] <= threshold:
                o[nb_features + feature] = threshold
            else:
                o[feature] = threshold
            a = policy[tuple(o.tolist() + [H])][zeta]
            # print(s, o, a)
        preds.append(a)
        lengths[i] = H
    return 1 - mean_squared_error(Y,preds), lengths.mean()

class State:
    def __init__(self, label: np.ndarray, nz: np.ndarray, is_terminal: bool = False):
        self.obs = label
        self.actions = []
        self.qs = []
        self.is_terminal = is_terminal
        self.nz = nz

    def add_action(self, action):
        self.actions.append(action)


class Action:
    def __init__(self, action: Union[np.ndarray, np.uint8]):
        self.action = action
        self.rewards = []
        self.probas = []
        self.next_states = []

    def transition(self, reward: float, proba: float, next_s: State):
        self.rewards.append(reward)
        self.probas.append(proba)
        self.next_states.append(next_s)


def build_mdp(
    S: np.ndarray,
    Y: np.ndarray,
    max_depth: int,
    aig_fn: AIGSelector,
    preselect_last_depth=True,
    verbose: bool = False,
):
    max_depth += 1
    # MAX 256 classes !!!!!
    data = Data(np.array(S, dtype=np.float64), np.array(Y, dtype=np.uint8))
    aig_fn.data = data
    nb_features = data.x.shape[1]
    root = State(
        np.concatenate(
            (data.x.min(axis=0) - 1e-3, data.x.max(axis=0) + 1e-3), dtype=np.float64
        ),
        nz=np.ones(data.x.shape[0], dtype=np.bool_),
    )
    terminal_state = np.zeros(2 * nb_features, dtype=np.float64)
    deci_nodes = [[root]]
    d = 0

    while d < max_depth:
        tmp = []
        for node in deci_nodes[d]:
            obs = node.obs.copy()
            expand_node = False
            classes, counts = np.unique(data.y[node.nz], return_counts=True)
            # If there is still depth budget and the current split has more than 1 class:
            if (d + 1) < max_depth and classes.shape[0] >= 2:
                if verbose:
                    print("Depth: {}, Nodes: {}".format(d, len(tmp)))
                expand_node = True
                if preselect_last_depth:
                    if d == max_depth - 2:
                        (
                            feat_thresh,
                            lefts,
                            rights,
                            probas_left,
                            probas_right,
                        ) = last_depth_select(data.x, data.y, node.nz)
                    else:
                        (
                            feat_thresh,
                            lefts,
                            rights,
                            probas_left,
                            probas_right,
                        ) = aig_fn.select(node.nz)
                else:
                    (
                        feat_thresh,
                        lefts,
                        rights,
                        probas_left,
                        probas_right,
                    ) = aig_fn.select(node.nz)

            rstar = max(counts) / node.nz.sum() - 1.0
            astar = classes[np.argmax(counts)]
            next_state = State(terminal_state, [0], is_terminal=True)
            next_state.qs = [rstar]
            a = Action(astar)
            a.transition(rstar, 1, next_state)
            node.add_action(a)
            if expand_node:
                for i, split in enumerate(feat_thresh):
                    a = Action(split)
                    feature, threshold = split
                    next_obs_left = obs.copy()
                    next_obs_left[nb_features + feature] = threshold
                    next_obs_right = obs.copy()
                    next_obs_right[feature] = threshold

                    if lefts[i].sum() > 0:
                        next_state_left = State(next_obs_left, lefts[i])
                        a.transition(0, probas_left[i], next_state_left)
                        tmp.append(next_state_left)
                    if rights[i].sum() > 0:
                        next_state_right = State(next_obs_right, rights[i])
                        a.transition(0, probas_right[i], next_state_right)
                        tmp.append(next_state_right)
                    if a.rewards != []:
                        node.add_action(a)

        if tmp != []:
            deci_nodes.append(tmp)
            d += 1
        else:
            break
    return deci_nodes




def build_mdp_regression(
    S: np.ndarray,
    Y: np.ndarray,
    max_depth: int,
    aig_fn: AIGSelector,
    preselect_last_depth=False,
    verbose: bool = False,
):
    max_depth += 1
    data = Data(np.array(S, dtype=np.float64), np.array(Y, dtype=np.float64))
    aig_fn.data = data
    nb_features = data.x.shape[1]
    root = State(
        np.concatenate(
            (data.x.min(axis=0) - 1e-3, data.x.max(axis=0) + 1e-3), dtype=np.float64
        ),
        nz=np.ones(data.x.shape[0], dtype=np.bool_),
    )
    terminal_state = np.zeros(2 * nb_features, dtype=np.float64)
    deci_nodes = [[root]]
    d = 0

    while d < max_depth:
        tmp = []
        for node in deci_nodes[d]:
            obs = node.obs.copy()
            expand_node = False
            # classes, counts = np.unique(data.y[node.nz], return_counts=True)
            # If there is still depth budget and the current split has more than 1 class:
            if (d + 1) < max_depth:
                if verbose:
                    print("Depth: {}, Nodes: {}".format(d, len(tmp)))
                expand_node = True
                if preselect_last_depth:
                    if d == max_depth - 2:
                        (
                            feat_thresh,
                            lefts,
                            rights,
                            probas_left,
                            probas_right,
                        ) = last_depth_select(data.x, data.y, node.nz)
                    else:
                        (
                            feat_thresh,
                            lefts,
                            rights,
                            probas_left,
                            probas_right,
                        ) = aig_fn.select(node.nz)
                else:
                    (
                        feat_thresh,
                        lefts,
                        rights,
                        probas_left,
                        probas_right,
                    ) = aig_fn.select(node.nz)

            astar = data.y[node.nz].mean(axis=0)
            rstar = - mean_squared_error(data.y[node.nz], data.y[node.nz].shape[0] * [astar])
            next_state = State(terminal_state, [0], is_terminal=True)
            next_state.qs = [rstar]
            a = Action(astar)
            a.transition(rstar, 1, next_state)
            node.add_action(a)
            if expand_node:
                for i, split in enumerate(feat_thresh):
                    a = Action(split)
                    feature, threshold = split
                    next_obs_left = obs.copy()
                    next_obs_left[nb_features + feature] = threshold
                    next_obs_right = obs.copy()
                    next_obs_right[feature] = threshold

                    if lefts[i].sum() > 0:
                        next_state_left = State(next_obs_left, lefts[i])
                        a.transition(0, probas_left[i], next_state_left)
                        tmp.append(next_state_left)
                    if rights[i].sum() > 0:
                        next_state_right = State(next_obs_right, rights[i])
                        a.transition(0, probas_right[i], next_state_right)
                        tmp.append(next_state_right)
                    if a.rewards != []:
                        node.add_action(a)

        if tmp != []:
            deci_nodes.append(tmp)
            d += 1
        else:
            break
    return deci_nodes
