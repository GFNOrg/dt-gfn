# Enable Cython optimizations
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport log, exp, lgammaf 
from libc.stdlib cimport malloc, free
from cython.parallel import prange

cdef class Attribute:
    TYPE = 0
    FEATURE = 1
    THRESHOLD = 2
    PROBABILITY = 3
    ACTIVE = 4
    N = 5

cdef int ATTRIBUTE_TYPE = Attribute.TYPE
cdef int ATTRIBUTE_FEATURE = Attribute.FEATURE
cdef int ATTRIBUTE_THRESHOLD = Attribute.THRESHOLD
cdef int ATTRIBUTE_PROBABILITY = Attribute.PROBABILITY
cdef int ATTRIBUTE_ACTIVE = Attribute.ACTIVE
cdef int ATTRIBUTE_N = Attribute.N

cdef class NodeType:
    CONDITION = 0
    CLASSIFIER = 1

cdef int NODE_TYPE_CONDITION = NodeType.CONDITION
cdef int NODE_TYPE_CLASSIFIER = NodeType.CLASSIFIER

ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def predict_proba_cython(
    np.ndarray[np.float32_t, ndim=2] state,
    np.ndarray[np.float32_t, ndim=1] x,
    np.ndarray[np.int32_t, ndim=1] node_types,
    np.ndarray[np.int32_t, ndim=1] features,
    np.ndarray[np.float32_t, ndim=1] thresholds,
    np.ndarray[np.float32_t, ndim=1] probabilities,
    int k,
    bint dirichlet,
    int attribute_n
):
    cdef int node_index = k
    cdef np.ndarray[np.float32_t, ndim=1] proba

    while node_types[node_index] == 0:
        if x[features[node_index]] < thresholds[node_index]:
            node_index = 2 * node_index + 1  # Left child
        else:
            node_index = 2 * node_index + 2  # Right child

    if dirichlet:
        proba = state[node_index, attribute_n:]
    else:
        proba = np.array([probabilities[node_index]], dtype=np.float32)

    return proba, node_index

@cython.boundscheck(False)
@cython.wraparound(False)
def traverse_tree_cython(np.ndarray[np.float32_t, ndim=2] state,
                         np.ndarray[np.float32_t, ndim=1] x,
                         int k=0):
    cdef int n_nodes = state.shape[0]
    cdef int feature_idx
    cdef float threshold

    while True:
        if state[k, Attribute.TYPE] == NodeType.CLASSIFIER:
            return k

        feature_idx = int(state[k, 1])
        threshold = state[k, 2]

        if x[feature_idx] < threshold:
            k = 2 * k + 1
        else:
            k = 2 * k + 2

        if k >= n_nodes:
            raise ValueError("Invalid tree structure")

@cython.boundscheck(False)
@cython.wraparound(False)
def find_leaves_cython(np.ndarray[np.float32_t, ndim=2] state):
    cdef int n_nodes = state.shape[0] - 1  # Exclude the last row
    cdef list leaves = []
    cdef int i

    for i in range(n_nodes):
        if state[i, Attribute.TYPE] == NodeType.CLASSIFIER:
            leaves.append(i)

    return leaves

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_log_likelihood_cython(state, samples, labels, alpha, int n_classes):
    cdef np.ndarray[np.float32_t, ndim=2] state_np
    cdef np.ndarray[np.float32_t, ndim=2] samples_np
    cdef np.ndarray[np.int32_t, ndim=1] labels_np
    cdef np.ndarray[np.float32_t, ndim=1] alpha_np

    state_np = np.asarray(state, dtype=np.float32)
    samples_np = np.asarray(samples, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int32)
    alpha_np = np.asarray(alpha, dtype=np.float32)

    cdef int n_samples = samples_np.shape[0]
    cdef int n_leaves = 0
    cdef int i, j, node
    cdef float log_likelihood = 0.0
    cdef np.ndarray[np.int32_t, ndim=2] leaf_counts
    cdef np.ndarray[np.int32_t, ndim=1] leaf_total_flow
    cdef list state_leaves
    
    state_leaves = find_leaves_cython(state_np)
    n_leaves = len(state_leaves)
    
    leaf_counts = np.zeros((n_leaves, n_classes), dtype=np.int32)
    leaf_total_flow = np.zeros(n_leaves, dtype=np.int32)

    for i in range(n_samples):
        node = traverse_tree_cython(state_np, samples_np[i])
        leaf_index = state_leaves.index(node)
        leaf_total_flow[leaf_index] += 1
        leaf_counts[leaf_index, labels_np[i]] += 1

    for i in range(n_leaves):
        log_likelihood += log_dirichlet_cython(leaf_counts[i].astype(np.float32) + alpha_np) - log_dirichlet_cython(alpha_np)

    return n_leaves, log_likelihood

@cython.cdivision(True)
cdef float log_dirichlet_cython(np.ndarray[np.float32_t, ndim=1] params):
    cdef float result = 0.0
    cdef float sum_params = 0.0
    cdef int i, n = params.shape[0]

    for i in range(n):
        result += lgammaf(params[i])
        sum_params += params[i]

    result -= lgammaf(sum_params)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def batch_predict_proba_cython(np.ndarray[np.float32_t, ndim=2] state,
                                np.ndarray[np.float32_t, ndim=2] X,
                                np.ndarray[np.int32_t, ndim=1] node_types,
                                np.ndarray[np.int32_t, ndim=1] features,
                                np.ndarray[np.float32_t, ndim=1] thresholds,
                                np.ndarray[np.float32_t, ndim=2] probabilities,
                                int attribute_n):
    cdef int n_samples = X.shape[0]
    cdef int n_nodes = state.shape[0]
    cdef int n_outputs = max(1, len(state[0, attribute_n:]))
    cdef np.ndarray[np.float32_t, ndim=2] predictions = np.zeros((n_samples, n_outputs), dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] node_indices = np.zeros(n_samples, dtype=np.int32)
    cdef int i, k, feature_idx
    cdef float threshold

    for i in range(n_samples):
        k = 0
        while True:
            if state[k, Attribute.TYPE] == NodeType.CLASSIFIER:
                if len(state[0, attribute_n:]) > 0:
                    predictions[i] = state[k, attribute_n:]
                else:
                    predictions[i] =  np.array([probabilities[0][k]])
                node_indices[i] = k
                break

            feature_idx = features[k]
            threshold = thresholds[k]

            if X[i, feature_idx] < threshold:
                k = 2 * k + 1
            else:
                k = 2 * k + 2

            if k >= n_nodes:
                raise ValueError("Invalid tree structure")

    return predictions, node_indices

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_predict_proba_multiple_states_cython(np.ndarray[np.float32_t, ndim=3] states,
                                            np.ndarray[np.float32_t, ndim=2] X,
                                            np.ndarray[np.int32_t, ndim=2] node_types,
                                            np.ndarray[np.int32_t, ndim=2] features,
                                            np.ndarray[np.float32_t, ndim=2] thresholds,
                                            np.ndarray[np.float32_t, ndim=3] probabilities,
                                            int attribute_n):
    cdef int n_states = states.shape[0]
    cdef int n_samples = X.shape[0]
    cdef int n_nodes = states.shape[1]
    cdef int n_outputs = max(1, len(states[0, 0, attribute_n:]))
    cdef np.ndarray[np.float32_t, ndim=3] predictions = np.zeros((n_states, n_samples, n_outputs), dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=2] node_indices = np.zeros((n_states, n_samples), dtype=np.int32)
    cdef int s, i, k, feature_idx
    cdef float threshold

    for s in range(n_states):
        for i in range(n_samples):
            k = 0
            while True:
                if states[s, k, Attribute.TYPE] == NodeType.CLASSIFIER:
                    if len(states[s, 0, attribute_n:]) > 0:
                        predictions[s, i] = states[s, k, attribute_n:]
                    else:
                        predictions[s, i] = np.array([probabilities[s, 0, k]])
                    node_indices[s, i] = k
                    break

                feature_idx = features[s, k]
                threshold = thresholds[s, k]

                if X[i, feature_idx] < threshold:
                    k = 2 * k + 1
                else:
                    k = 2 * k + 2

                if k >= n_nodes:
                    raise ValueError("Invalid tree structure")

    return predictions, node_indices

@cython.boundscheck(False)
@cython.wraparound(False)
def get_mask_invalid_actions_forward_cy(
    np.ndarray[np.float32_t, ndim=2] state,
    np.ndarray[np.int64_t, ndim=0] stage,  # torch.Tensor of integer type
    int n_nodes,
    int action_space_dim,
    int action_index_pick_leaf,
    int action_index_eos,
    int action_index_pick_leaf_probability,
    int action_index_pick_feature,
    int action_index_pick_threshold,
    int action_index_pick_left_child_probability,
    int action_index_pick_right_child_probability,
    int action_index_pick_triplet,
    bint continuous,
    bint mask_redundant_choices,
    np.ndarray[np.float64_t, ndim=1] thresholds  # Note: float64 for thresholds
):
    cdef np.ndarray[np.uint8_t, ndim=1] mask = np.ones(action_space_dim, dtype=np.uint8)
    cdef int k, idx, feature_chosen
    cdef float node_feature, node_threshold, current_threshold
    
    if stage == 0:  # COMPLETE
        leaves = np.where(state[:, 0] == 1)[0]  # Assuming 1 represents CLASSIFIER
        for k in leaves:
            if (k * 2 + 2) < n_nodes:
                mask[action_index_pick_leaf + k] = 0
        mask[action_index_eos] = 0
    elif stage == 1:  # LEAF
        k = np.where(state[:, 4] == 1)[0][0]  # Find active node
        if continuous:
            mask[action_index_pick_leaf_probability] = 0
        else:
            prob_idx = int(state[k, 3] * 100)  # Assuming probability is stored * 100
            mask[action_index_pick_leaf_probability + prob_idx] = 0
    elif stage == 2:  # LEAF_PROBABILITY
        for idx in range(action_index_pick_feature, action_index_pick_threshold):
            mask[idx] = 0
    elif stage == 3:  # FEATURE
        for idx in range(action_index_pick_threshold, action_index_pick_left_child_probability):
            mask[idx] = 0
        
        if mask_redundant_choices:
            k = np.where(state[:, 4] == 1)[0][0]  # Find active node
            feature_chosen = int(state[k, 1])
            path = []
            while k != 0:
                node_feature = state[k, 1]
                if node_feature == feature_chosen:
                    node_threshold = state[k, 2]
                    direction = 'left' if k % 2 == 1 else 'right'
                    path.append((node_threshold, direction))
                k = (k - 1) // 2
            
            if path:
                for idx in range(action_index_pick_threshold + 1, action_index_pick_left_child_probability - 1):
                    current_threshold = thresholds[idx - action_index_pick_threshold]
                    for threshold, direction in path:
                        if threshold < 0:
                            continue
                        if (direction == 'left' and current_threshold >= threshold) or \
                           (direction == 'right' and current_threshold <= threshold) or \
                           (current_threshold == threshold):
                            mask[idx] = 1
                            break
    elif stage == 4:  # THRESHOLD
        for idx in range(action_index_pick_left_child_probability, action_index_pick_right_child_probability):
            mask[idx] = 0
    elif stage == 5:  # LEFT_CHILD_PROBABILITY
        for idx in range(action_index_pick_right_child_probability, action_index_pick_triplet):
            mask[idx] = 0
    elif stage == 6:  # RIGHT_CHILD_PROBABILITY
        k = np.where(state[:, 4] == 1)[0][0]  # Find active node
        mask[action_index_pick_triplet + k] = 0
    
    return mask