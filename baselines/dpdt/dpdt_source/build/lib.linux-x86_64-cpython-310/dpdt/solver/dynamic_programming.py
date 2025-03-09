import time

import numpy as np

from dpdt.utils.feature_selectors import AIGSelector
from dpdt.utils.mdp import State, build_mdp, eval_in_mdp
from dpdt.utils.tree import extract_tree

import pickle


def backward_induction(mdp: list[list[State]], zeta: float):
    # can be done when building mdp by just storing vectors of r.
    for d in mdp:
        for node in d:
            for action in node.actions:
                for i, next_state in enumerate(action.next_states):
                    if not next_state.is_terminal:
                        action.rewards[i] = zeta

    policy = dict()
    max_depth = len(mdp)
    for H, d in enumerate(reversed(mdp)):
        for node in d:
            best_a = None
            best_qa = float("-inf")
            for a in node.actions:
                q_s_a = 0
                for j, s_next in enumerate(a.next_states):
                    q_s_a += a.probas[j] * (a.rewards[j] + max(s_next.qs))
                node.qs.append(q_s_a)
                if q_s_a > best_qa:
                    best_qa = q_s_a
                    best_a = a
            policy[tuple(node.obs.tolist() + [max_depth - H - 1])] = best_a.action
    return policy


def backward_induction_multiple_zetas(mdp: list[list[State]], zetas: np.ndarray):
    policy = dict()
    max_depth = len(mdp)
    for H, d in enumerate(reversed(mdp)):
        for node in d:
            qs = []
            for a in node.actions:
                if isinstance(a.action, np.uint8) or isinstance(a.action, np.ndarray):
                    q_s_a = np.ones(zetas.shape[0]) * a.rewards[0]
                else:
                    q_s_a = np.zeros(zetas.shape[0])
                    for j, s_next in enumerate(a.next_states):
                        q_s_a += a.probas[j] * (zetas + s_next.v)
                qs.append(q_s_a)
            qs = np.asarray(qs)
            argmax_qs = np.argmax(qs, axis=0)
            node.v = np.take_along_axis(qs, argmax_qs[None, :], 0)[0]
            policy[tuple(node.obs.tolist() + [max_depth - H - 1])] = [
                node.actions[k].action for k in argmax_qs
            ]
    return policy

def count_states_f(mdp: list[list[State]]):
    nb = 0
    for H, d in enumerate(reversed(mdp)):
        for node in d:
            nb += 1
    return nb


def dpdt(
    S: np.ndarray,
    Y: np.ndarray,
    aig_fn: AIGSelector,
    zetas: np.ndarray,
    max_depth: int = 2,
    preselect_last_depth=True,
    save_policy=False,
    verbose: bool = False,
    plot_tree: bool = False,
    tree_folder: str = "",
    policy_folder: str = "saved_policies",
    count_states: bool = False,
):
    if verbose:
        print("Building MDP")
    start = time.time()
    tree = build_mdp(S, Y, max_depth, aig_fn, preselect_last_depth, verbose)
    if count_states:
        nb_states = count_states_f(tree)
        print(nb_states, "STATES")
    if verbose:
        print("Backward")
    policy = backward_induction_multiple_zetas(tree, zetas)
    end = time.time()
    time_ = end - start
    if save_policy:
        with open(policy_folder + ".pkl", "wb") as f:
            pickle.dump(policy, f)
    # To Do add save mdps here
    init_obs = tree[0][0].obs
    scores, depths, nodes = (
        np.zeros(zetas.shape[0], dtype=np.float64),
        np.zeros(zetas.shape[0], dtype=np.uint8),
        np.zeros(zetas.shape[0], dtype=np.uint8),
    )
    if verbose:
        print("Eval Policy")
    for i, zeta in enumerate(zetas):
        scores[i] = eval_in_mdp(S, Y, policy, init_obs, i)
        tree_, nodes[i], depths[i] = extract_tree(policy, init_obs, zeta=i)
        if plot_tree:
            graph = tree_.graphviz()
            graph.body
            graph.render(tree_folder + "arbre" + str(nodes[i])+ "_" + str(round(scores[i],4)) , format="png")
    if count_states:
        return scores, depths, nodes, time_, nb_states
    return scores, depths, nodes, time_
