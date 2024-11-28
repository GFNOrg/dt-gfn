import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="A single label was found in 'y_true' and 'y_pred'.",
)

import code
import multiprocessing
import pickle
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.tree_acc_cython import (
    batch_predict_proba_cython, batch_predict_proba_multiple_states_cython,
    get_mask_invalid_actions_forward_cy, predict_proba_cython)
from gflownet.utils.common import (convert_sklearn_tree_to_custom_state,
                                   hash_tensor)
from networkx.drawing.nx_pydot import graphviz_layout
from numba import boolean, float32, int64, jit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from torch.distributions import (Beta, Categorical, Dirichlet,
                                 MixtureSameFamily, Uniform)
from torchtyping import TensorType
from tqdm import tqdm


class NodeType:
    """
    Encodes two types of nodes present in a tree:
    0 - condition node (node other than leaf), that stores the information
        about on which feature to make the decision, and using what threshold.
    1 - classifier node (leaf), that stores the information about the positive
        class probability that will be predicted once that node is reached.
    """

    CONDITION = 0
    CLASSIFIER = 1


class Status:
    """
    Status of the node. Every node except the one on which a macro step
    was initiated will be marked as inactive; the node will be marked
    as active iff the process of its splitting is in progress.
    """

    INACTIVE = 0
    ACTIVE = 1


class Stage:
    """
    Current stage of the tree, encoded as part of the state.
    0 - complete, indicates that there is no macro step initiated, and
        the only allowed action is to pick one of the leaves for splitting.
    1 - leaf, indicates that a leaf was picked for splitting, and the only
        allowed action is a "dummy" action (in the forward mode; meaningful
        when going backward) of selecting the probability currently stored
        in the leaf selected for splitting.
    2 - leaf probability, indicates that the leaf probability was picked, and
        the only allowed action is picking a feature on which it will be split.
    3 - feature, indicates that a feature was picked, and the only allowed
        action is picking a threshold for splitting.
    4 - threshold, indicates that a threshold was picked, and the only
        allowed action is picking the class probability of left child.
    5 - left probability, indicates that class probability of the left child
        was picked, and the only allowed action is picking the class
        probability of the right child.
    6 - right probability, indicates that class probability of the right child
        was picked, and the only allowed action is the "dummy" action (in the
        forward mode; meaningful when going backward) of selecting a triplet
        of (parent, left child, right child).
    """

    COMPLETE = 0
    LEAF = 1
    LEAF_PROBABILITY = 2
    FEATURE = 3
    THRESHOLD = 4
    LEFT_CHILD_PROBABILITY = 5
    RIGHT_CHILD_PROBABILITY = 6


class ActionType:
    """
    Type of action that will be passed to Tree.step. Refer to Stage for details.
    """

    PICK_LEAF = 0
    PICK_LEAF_PROBABILITY = 1
    PICK_FEATURE = 2
    PICK_THRESHOLD = 3
    PICK_LEFT_CHILD_PROBABILITY = 4
    PICK_RIGHT_CHILD_PROBABILITY = 5
    PICK_TRIPLET = 6


class Attribute:
    """
    Contains indices of individual attributes in a state tensor.

    Types of attributes defining each node of the tree:

        0 - node type (condition or classifier),
        1 - index of the feature used for splitting (condition node only, -1 otherwise),
        2 - decision threshold (condition node only, -1 otherwise),
        3 - probability output (classifier node only, -1 otherwise),
        4 - whether the node has active status (1 if node was picked and the macro step
            didn't finish yet, 0 otherwise).
    """

    TYPE = 0
    FEATURE = 1
    THRESHOLD = 2
    PROBABILITY = 3
    ACTIVE = 4
    N = 5  # Total number of attributes.


class Tree(GFlowNetEnv):
    """
    GFlowNet environment representing a decision tree.

    Constructing a tree consists of a combination of macro steps (picking a leaf
    to split using a given feature, threshold, and children probabilities), which are divided
    into a series of consecutive micro steps (1 - pick a leaf, 2 - (dummy) pick leaf
    probability, 3 - pick a feature, 4 - pick a threshold, 5 - pick a positive class
    probability of the left child, 6 - pick a positive class probability of the right child),
    7 - (dummy) pick triplet of (parent, left child, right child). A consequence
    of that is, as long as a macro step is not in progress, the tree constructed so
    far is always a valid decision tree, which means that forward-looking loss etc. can
    be used.

    Two dummy actions are introduced in the forward step, because they are necessary to
    correctly model backward transitions. Specifically, which actions are meaningful in
    either forward or backward step was described below:

    ACTIONS                   | FORWARD | BACKWARD
    -----------------------------------------------
    select leaf               |    ✓    |    ✗    |
    select leaf probability   |    ✗    |    ✓    |
    select feature            |    ✓    |    ✗    |
    select feature threshold  |    ✓    |    ✗    |
    select left probability   |    ✓    |    ✗    |
    select right probability  |    ✓    |    ✗    |
    select triplet            |    ✗    |    ✓    |

    Internally, the tree is represented as a fixed-shape tensor (thus, specifying
    the maximum depth is required), with nodes indexed from k = 0 to 2**max_depth - 2,
    and each node containing a 5-element attribute tensor (see class Attribute for
    details). The nodes are indexed from top left to bottom right, as follows:

                0
        1               2
    3       4       5       6

    States are represented by a tensor with shape [n_nodes + 1, 5], where each k-th row
    corresponds to the attributes of the k-th node of the tree. The last row contains
    the information about the stage of the tree (see class Stage).
    """

    def __init__(
        self,
        X_train: Optional[npt.NDArray] = None,
        y_train: Optional[npt.NDArray] = None,
        X_test: Optional[npt.NDArray] = None,
        y_test: Optional[npt.NDArray] = None,
        data_path: Optional[str] = None,
        scale_data: bool = True,
        max_depth: int = 10,
        continuous: bool = False,
        beta_binomial: bool = False,
        dirichlet: bool = False,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.0,
        n_thresholds: Optional[int] = 99,
        n_quantiles: Optional[int] = None,
        n_probabilities: Optional[int] = 9,
        n_classes: Optional[int] = None,
        prior: Optional[List[float]] = None,
        threshold_components: int = 1,
        beta_params_min: float = 0.1,
        beta_params_max: float = 2.0,
        fixed_distr_params: dict = {
            "beta_alpha": 2.0,
            "beta_beta": 5.0,
        },
        random_distr_params: dict = {
            "beta_alpha": 1.0,
            "beta_beta": 1.0,
        },
        policy_format: str = "mlp",
        test_args: dict = {"top_k_trees": 0},
        mask_redundant_choices: bool = True,
        **kwargs,
    ):
        """
        Attributes
        ----------
        X_train : np.array
            Train dataset, with dimensionality (n_observations, n_features). It may be
            None if a data set is provided via data_path.

        y_train : np.array
            Train labels, with dimensionality (n_observations,). It may be
            None if a data set is provided via data_path.

        X_test : np.array
            Test dataset, with dimensionality (n_observations, n_features). It may be
            None if a data set is provided via data_path, or if you don't want to perform
            test set evaluation.

        y_train : np.array
            Test labels, with dimensionality (n_observations,). It may be
            None if a data set is provided via data_path, or if you don't want to perform
            test set evaluation.

        data_path : str
            A path to a data set, with the following options:
            - *.pkl: Pickled dict with X_train, y_train, and (optional) X_test and y_test
              variables.
            - *.csv: CSV containing an optional 'Split' column in the last place, containing
              'train' and 'test' values, and M remaining columns, where the first (M - 1)
              columns will be taken to construct the input X, and M-th column will be the
              target y.
            Ignored if X_train and y_train are not None.

        scale_data : bool
            Whether to perform min-max scaling on the provided data (to a [0; 1] range).

        max_depth : int
            Maximum depth of a tree.

        continuous : bool
            Whether the environment should operate in a continuous mode (in which distribution
            parameters are predicted for the threshold) or the discrete mode (in which there
            is a discrete set of possible thresholds to choose from).

        beta_binomial : bool
            Whether the environment should operate as if beta-binomial reward is used,
            that is instead of having separate actions for picking probabilities, there are
            dummy actions that set it to -1, and during evaluation they are filled by sampling
            directly from beta-binomial based on the dataset.

        dirichlet : bool
            Whether the environment should operate as if Dirichlet reward is used,
            that is instead of having separate actions for picking probabilities, there are
            dummy actions that set it to -1, and during evaluation they are filled by sampling
            directly from dirichlet based on the dataset.

        beta_alpha : float
            Alpha parameter for sampling probabilities from beta-binomial distribution.

        beta_beta : float
            Beta parameter for sampling probabilities from beta-binomial distribution.

        n_thresholds : int
            Number of uniformly distributed thresholds in a (0; 1) range that will be used
            in the discrete mode. Ignored if continuous is True.

        n_probabilities : int
            Number of uniformly distributed probabilities in a (0; 1) range that will be used
            in the discrete mode. Ignored if continuous is True.

        threshold_components : int
            The number of mixture components that will be used for sampling
            the threshold.

        policy_format : str
            Type of policy that will be used with the environment, either 'mlp' or 'gnn'.
            Influences which state2policy functions will be used.

        mask_redundant_choices : bool
            Masking (feature, threshold) tuples not splitting the hypothesis space any further.
        """
        if X_train is not None and y_train is not None:
            self.X_train = X_train
            self.y_train = y_train
            if X_test is not None and y_test is not None:
                self.X_test = X_test
                self.y_test = y_test
            else:
                self.X_test = None
                self.y_test = None
        elif data_path is not None:
            self.X_train, self.y_train, self.X_test, self.y_test = Tree._load_dataset(
                data_path
            )
        else:
            raise ValueError(
                "A Tree must be initialised with a data set. X_train, y_train and data_path cannot "
                "be all None"
            )
        if scale_data:
            self.scaler = MinMaxScaler().fit(self.X_train)
            self.X_train = self.scaler.transform(self.X_train)
            if self.X_test is not None:
                self.X_test = self.scaler.transform(self.X_test)
        self.y_train = self.y_train.astype(int)
        if self.y_test is not None:
            self.y_test = self.y_test.astype(int)
        assert not (beta_binomial and dirichlet)
        self.beta_binomial = beta_binomial
        self.dirichlet = dirichlet
        if not dirichlet and not set(self.y_train).issubset({0, 1}):
            raise ValueError(
                f"Expected y_train to have values in {{0, 1}}, received {set(self.y_train)}."
            )
        self.n_classes = n_classes if n_classes is not None else len(set(self.y_train))
        self.n_features = self.X_train.shape[1]
        self.max_depth = max_depth
        self.continuous = continuous
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.prior = prior
        if not continuous:
            if n_quantiles:
                self.thresholds = np.quantile(
                    self.X_train, np.linspace(0, 1, n_quantiles + 2)[1:-1]
                )
            else:
                self.thresholds = np.linspace(0, 1, n_thresholds + 2)[1:-1]
            self.thresh2idx = {
                np.round(p, 2): idx for idx, p in enumerate(self.thresholds)
            }
            if self.beta_binomial or self.dirichlet:
                self.probabilities = np.array([-1.0])
            else:
                self.probabilities = np.linspace(0, 1, n_probabilities + 2)[1:-1]
            self.prob2idx = {
                np.round(p, 2): idx for idx, p in enumerate(self.probabilities)
            }
            assert len(self.probabilities) == len(self.prob2idx)
        self.test_args = test_args
        self.mask_redundant_choices = mask_redundant_choices
        # Parameters of the policy distribution
        self.components = threshold_components
        self.beta_params_min = beta_params_min
        self.beta_params_max = beta_params_max
        # Source will contain information about the current stage (on the last position),
        # and up to 2**max_depth - 1 nodes, each with Attribute.N attributes, for a total of
        # 1 + Attribute.N * (2**max_depth - 1) values. The root (0-th node) of the
        # source is initialized with a classifier.
        self.n_nodes = 2**max_depth - 1
        self.source = torch.full((self.n_nodes + 1, Attribute.N), torch.nan)
        self._set_stage(Stage.COMPLETE, self.source)
        attributes_root = self.source[0]
        attributes_root[Attribute.TYPE] = NodeType.CLASSIFIER
        attributes_root[Attribute.FEATURE] = -1
        attributes_root[Attribute.THRESHOLD] = -1
        if self.beta_binomial or self.dirichlet:
            self.default_probability = -1
        elif self.continuous:
            self.default_probability = 0.5
        else:
            self.default_probability = self.probabilities[n_probabilities // 2]
        attributes_root[Attribute.PROBABILITY] = self.default_probability
        attributes_root[Attribute.ACTIVE] = Status.INACTIVE

        # End-of-sequence action.
        self.eos = (-1, -1)

        # Conversions
        policy_format = policy_format.lower()
        if policy_format == "mlp" or policy_format == "tree_mlp":
            self.states2policy = self.states2policy_mlp
        elif policy_format == "set_transformer":
            self.states2policy = self.states2policy_transformer
        elif policy_format != "gnn":
            raise ValueError(
                f"Unrecognized policy_format = {policy_format}, expected either 'mlp', 'gnn' or 'set_transformer'."
            )

        super().__init__(
            fixed_distr_params=fixed_distr_params,
            random_distr_params=random_distr_params,
            continuous=continuous,
            **kwargs,
        )

    @staticmethod
    def _get_parent(k: int) -> Optional[int]:
        """
        Get node index of a parent of k-th node.
        """
        return (k - 1) >> 1 if k > 0 else None

    @staticmethod
    def _get_left_child(k: int) -> int:
        """
        Get node index of a left child of k-th node.
        """
        return (k << 1) + 1

    @staticmethod
    def _get_right_child(k: int) -> int:
        """
        Get node index of a right child of k-th node.
        """
        return (k << 1) + 2

    @staticmethod
    def _get_sibling(k: int) -> Optional[int]:
        """
        Get node index of the sibling of k-th node.
        """
        parent = Tree._get_parent(k)
        if parent is None:
            return None
        left = Tree._get_left_child(parent)
        right = Tree._get_right_child(parent)
        return left if k == right else right

    def _get_stage(self, state: Optional[torch.Tensor] = None) -> int:
        """
        Returns the stage of the current environment from self.state[-1, 0] or from the
        state passed as an argument.
        """
        if state is None:
            state = self.state
        return state[-1, 0]

    def generate_random_tree(self):
        """
        Generates a decision tree (structure) randomly without redundant splits.
        """
        n_nodes = 2**self.max_depth - 1
        tree = np.full((n_nodes + 1, Attribute.N), np.nan, dtype=float)

        def recursive_split(node, depth, used_splits):
            if depth >= self.max_depth - 1 or np.random.random() < 0.3:  # Leaf node
                tree[node, Attribute.TYPE] = 1
                tree[node, Attribute.FEATURE : Attribute.ACTIVE] = -1
            else:
                # Internal node
                feature = np.random.randint(0, self.n_features)
                threshold = np.random.choice(self.thresholds)

                # Ensure no redundant splits
                while (feature, threshold) in used_splits:
                    feature = np.random.randint(0, self.n_features)
                    threshold = np.random.choice(self.thresholds)

                tree[node, Attribute.TYPE] = 0
                tree[node, Attribute.FEATURE] = feature
                tree[node, Attribute.THRESHOLD] = threshold
                tree[node, Attribute.PROBABILITY] = -1
                tree[node, Attribute.ACTIVE] = 0

                # Add current split to the used splits
                used_splits.add((feature, threshold))

                left_child = 2 * node + 1
                right_child = 2 * node + 2

                # Recur for children with current path splits
                recursive_split(left_child, depth + 1, used_splits.copy())
                recursive_split(right_child, depth + 1, used_splits.copy())

        recursive_split(0, 0, set())

        # Set the last row (stage information)
        tree[-1, Attribute.TYPE] = 0
        return torch.tensor(tree, dtype=self.float)

    def get_random_unmasked_terminating_states(self, n, rf_proportion=0.0, seed=None):
        """
        Generates n terminating states, alternating between random trees and trees from a random forest.

        :param n: Total number of trees to generate
        :param rf_proportion: Proportion of trees to be generated from random forest (0 to 1)
        :param seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        n_rf_trees = int(n * rf_proportion)
        n_random_trees = n - n_rf_trees

        # Generate trees from random forest
        if n_rf_trees > 0:
            rf_states = self.get_terminating_states_from_random_forest(
                n_rf_trees, seed=seed
            )
        else:
            rf_states = []

        # Generate random trees
        random_states = [self.generate_random_tree() for _ in range(n_random_trees)]

        # Combine and shuffle
        combined_states = rf_states + random_states
        np.random.shuffle(combined_states)

        return combined_states

    def get_terminating_states_from_random_forest(self, n, seed=None):
        """
        Generates n random terminating states efficiently using a Random Forest.
        """
        if seed is not None:
            np.random.seed(seed)

        # Train a Random Forest
        rf = RandomForestClassifier(
            n_estimators=n, max_depth=self.max_depth - 1, random_state=seed
        )
        rf.fit(self.X_train, self.y_train)

        terminating_states = []

        for estimator in tqdm(rf.estimators_, desc="Generating RF terminating states"):
            state = torch.tensor(
                convert_sklearn_tree_to_custom_state(
                    estimator.tree_, self.max_depth, thresholds=self.thresholds
                ),
                dtype=self.float,
            )
            terminating_states.append(state.clone())

        return terminating_states[:n]

    # TODO: Add checks for when the state space is intractable to traverse.
    def get_all_terminating_states(
        self,
        state: Optional[torch.Tensor] = None,
        return_hash_keys: Optional[bool] = False,
    ) -> List[torch.Tensor]:
        """
        Returns a tensor of all terminating states using dynamic programming.
        """
        if self.continuous:
            raise ValueError(f"Task intractable with continuous values!")

        if state is None:
            state = self.state

        initial_state = state.clone()  # Store current state of the tree

        hashmap = {hash_tensor(state): state}
        max_depth = self.max_depth
        k = 8000

        # (self.n_features + len(self.probabilities) + len(self.thresholds)) * (
        #    2 ** (max_depth - 2) + 1
        # ) + 20  # 20 used as a bound confidence window

        # Explore all possible trajectories starting from a given input state, or the tree's initial state by default.
        while k > 0:
            h = hashmap.copy()
            for key in h:
                self.set_state(h[key])
                actions = self.get_valid_actions()
                for i in range(len(actions)):
                    self.set_state(h[key])
                    state, _, _ = self.step(actions[i])
                    if hash_tensor(state) not in hashmap:
                        hashmap[hash_tensor(state)] = state
            k -= 1
            if k % 100 == 0:
                print(k)

        valid_states = {}

        for key in hashmap:
            state = hashmap[key]
            if self._get_stage(state) == Stage.COMPLETE:
                valid_states[hash_tensor(state)] = state

        self.set_state(initial_state)  # Reset state to initial state

        if return_hash_keys:
            return valid_states
        else:
            return list(valid_states.values())

    def _set_stage(
        self, stage: int, state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sets the stage of the current environment (self.state) or of the state passed
        as an argument by updating state[-1, 0].
        """
        if state is None:
            state = self.state
        state[-1, 0] = stage
        return state

    def _pick_leaf(self, k: int) -> None:
        """
        Select one of the leaves (classifier nodes) that will be split, and initiate
        macro step.
        """
        attributes = self.state[k]

        assert self._get_stage() == Stage.COMPLETE
        assert attributes[Attribute.TYPE] == NodeType.CLASSIFIER
        assert not torch.any(torch.isnan(attributes))
        assert torch.all(attributes[1:3] == -1)
        if not (self.beta_binomial or self.dirichlet):
            assert attributes[Attribute.PROBABILITY] >= 0
        assert attributes[Attribute.ACTIVE] == Status.INACTIVE

        attributes[Attribute.TYPE] = NodeType.CONDITION
        attributes[Attribute.ACTIVE] = Status.ACTIVE

        self._set_stage(Stage.LEAF)

    def _pick_leaf_probability(self, k: int) -> None:
        """
        Dummy action that "picks" the probability currently stored in the
        active node.
        """
        attributes = self.state[k]

        assert self._get_stage() == Stage.LEAF
        assert attributes[Attribute.TYPE] == NodeType.CONDITION
        assert torch.all(attributes[1:3] == -1)
        if not (self.beta_binomial or self.dirichlet):
            assert attributes[Attribute.PROBABILITY] >= 0
        assert attributes[Attribute.ACTIVE] == Status.ACTIVE

        attributes[Attribute.PROBABILITY] = -1

        self._set_stage(Stage.LEAF_PROBABILITY)

    def _pick_feature(self, k: int, feature: float) -> None:
        """
        Select the feature on which currently selected leaf will be split.
        """
        attributes = self.state[k]

        assert self._get_stage() == Stage.LEAF_PROBABILITY
        assert attributes[Attribute.TYPE] == NodeType.CONDITION
        assert torch.all(attributes[1:4] == -1)
        assert attributes[Attribute.ACTIVE] == Status.ACTIVE

        attributes[Attribute.FEATURE] = feature

        self._set_stage(Stage.FEATURE)

    def _pick_threshold(self, k: int, threshold: float) -> None:
        """
        Select the threshold for splitting the currently selected leaf ond
        the selected feature.
        """
        attributes = self.state[k]

        assert self._get_stage() == Stage.FEATURE
        assert attributes[Attribute.TYPE] == NodeType.CONDITION
        assert attributes[Attribute.FEATURE] >= 0
        assert torch.all(attributes[2:4] == -1)
        assert attributes[Attribute.ACTIVE] == Status.ACTIVE

        attributes[Attribute.THRESHOLD] = threshold

        self._set_stage(Stage.THRESHOLD)

    def _pick_left_child_probability(self, k: int, probability: float) -> None:
        """
        Select the positive class probability for the left child of the
        currently selected leaf, given selected feature and threshold.
        """
        attributes = self.state[k]

        assert self._get_stage() == Stage.THRESHOLD
        assert attributes[Attribute.TYPE] == NodeType.CONDITION
        assert torch.all(attributes[1:3] >= 0)
        assert attributes[Attribute.PROBABILITY] == -1
        assert attributes[Attribute.ACTIVE] == Status.ACTIVE

        self._insert_classifier(Tree._get_left_child(k), probability=probability)

        self._set_stage(Stage.LEFT_CHILD_PROBABILITY)

    def _pick_right_child_probability(self, k: int, probability: float) -> None:
        """
        Select the positive class probability for the right child of the
        currently selected leaf, given selected feature and threshold, and
        finalize splitting.
        """
        attributes = self.state[k]

        assert self._get_stage() == Stage.LEFT_CHILD_PROBABILITY
        assert attributes[Attribute.TYPE] == NodeType.CONDITION
        assert torch.all(attributes[1:3] >= 0)
        assert attributes[Attribute.PROBABILITY] == -1
        assert attributes[Attribute.ACTIVE] == Status.ACTIVE
        assert (
            self.state[Tree._get_left_child(k)][Attribute.TYPE] == NodeType.CLASSIFIER
        )
        if not (self.beta_binomial or self.dirichlet):
            assert self.state[Tree._get_left_child(k)][Attribute.PROBABILITY] >= 0

        self._insert_classifier(Tree._get_right_child(k), probability=probability)

        self._set_stage(Stage.RIGHT_CHILD_PROBABILITY)

    def _pick_triplet(self, k: int) -> None:
        """
        Dummy action that "picks" the triplet of (parent, left child, right child)
        that was split in the current macro step.
        """
        attributes = self.state[k]

        assert self._get_stage() == Stage.RIGHT_CHILD_PROBABILITY
        assert attributes[Attribute.TYPE] == NodeType.CONDITION
        assert torch.all(attributes[1:3] >= 0)
        assert attributes[Attribute.PROBABILITY] == -1
        assert attributes[Attribute.ACTIVE] == Status.ACTIVE
        assert (
            self.state[Tree._get_left_child(k)][Attribute.TYPE] == NodeType.CLASSIFIER
        )
        assert (
            self.state[Tree._get_right_child(k)][Attribute.TYPE] == NodeType.CLASSIFIER
        )
        if not (self.beta_binomial or self.dirichlet):
            assert self.state[Tree._get_left_child(k)][Attribute.PROBABILITY] >= 0
            assert self.state[Tree._get_right_child(k)][Attribute.PROBABILITY] >= 0

        attributes[Attribute.ACTIVE] = Status.INACTIVE

        self._set_stage(Stage.COMPLETE)

    def _insert_classifier(self, k: int, probability: float) -> None:
        """
        Replace attributes of k-th node with those of a classifier node.
        """
        attributes = self.state[k]

        assert torch.all(torch.isnan(attributes))

        attributes[Attribute.TYPE] = NodeType.CLASSIFIER
        attributes[Attribute.FEATURE] = -1
        attributes[Attribute.THRESHOLD] = -1
        attributes[Attribute.PROBABILITY] = probability
        attributes[Attribute.ACTIVE] = Status.INACTIVE

    def get_action_space(self) -> List[Tuple[int, Union[int, float]]]:
        """
        Actions are a tuple containing:
            1) action type:
                0 - pick leaf to split,
                1 - pick leaf probability (dummy action in forward),
                2 - pick feature,
                3 - pick threshold,
                4 - pick left child probability,
                5 - pick right child probability,
                6 - pick triplet (dummy action in forward),
            2) action value, depending on the action type:
                pick leaf: node index,
                pick leaf probability: current probability output,
                pick feature: feature index,
                pick threshold: threshold value,
                pick left/right probability: probability value,
                pick triplet: node (triplet parent) index.
        """
        actions = []
        # Pick leaf
        self._action_index_pick_leaf = 0
        actions.extend([(ActionType.PICK_LEAF, idx) for idx in range(self.n_nodes)])
        # Pick leaf probability
        self._action_index_pick_leaf_probability = len(actions)
        if self.continuous:
            actions.append((ActionType.PICK_LEAF_PROBABILITY, -1))
        else:
            actions.extend(
                [
                    (ActionType.PICK_LEAF_PROBABILITY, idx)
                    for idx, _ in enumerate(self.probabilities)
                ]
            )
        # Pick feature
        self._action_index_pick_feature = len(actions)
        actions.extend(
            [(ActionType.PICK_FEATURE, idx) for idx in range(self.n_features)]
        )
        # Pick threshold
        self._action_index_pick_threshold = len(actions)
        if self.continuous:
            actions.append((ActionType.PICK_THRESHOLD, -1))
        else:
            actions.extend(
                [
                    (ActionType.PICK_THRESHOLD, idx)
                    for idx, _ in enumerate(self.thresholds)
                ]
            )
        # Pick left child probability
        self._action_index_pick_left_child_probability = len(actions)
        if self.continuous:
            actions.append((ActionType.PICK_LEFT_CHILD_PROBABILITY, -1))
        else:
            actions.extend(
                [
                    (ActionType.PICK_LEFT_CHILD_PROBABILITY, idx)
                    for idx, _ in enumerate(self.probabilities)
                ]
            )
        # Pick right child probability
        self._action_index_pick_right_child_probability = len(actions)
        if self.continuous:
            actions.append((ActionType.PICK_RIGHT_CHILD_PROBABILITY, -1))
        else:
            actions.extend(
                [
                    (ActionType.PICK_RIGHT_CHILD_PROBABILITY, idx)
                    for idx, _ in enumerate(self.probabilities)
                ]
            )
        # Pick triplet
        self._action_index_pick_triplet = len(actions)
        actions.extend([(ActionType.PICK_TRIPLET, idx) for idx in range(self.n_nodes)])
        # EOS
        self._action_index_eos = len(actions)
        actions.append(self.eos)

        return actions

    def step(
        self, action: Tuple[int, Union[int, float]], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int, Union[int, float]], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed.
            See: self.get_action_space()

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # Replace values of continuous actions by a representative value to allow
        # checking them.
        action_to_check = self.action2representative(action)
        do_step, self.state, action_to_check = self._pre_step(
            action_to_check,
            skip_mask_check=(skip_mask_check or self.skip_mask_check),
        )
        if not do_step:
            return self.state, action, False

        self.n_actions += 1

        if action != self.eos:
            action_type, action_value = action

            if action_type == ActionType.PICK_LEAF:
                self._pick_leaf(action_value)
            else:
                k = self.find_active(self.state)

                if action_type == ActionType.PICK_LEAF_PROBABILITY:
                    self._pick_leaf_probability(k)
                elif action_type == ActionType.PICK_FEATURE:
                    self._pick_feature(k, action_value)
                elif action_type == ActionType.PICK_THRESHOLD:
                    if self.continuous:
                        self._pick_threshold(k, action_value)
                    else:
                        self._pick_threshold(k, self.thresholds[action_value])
                elif action_type == ActionType.PICK_LEFT_CHILD_PROBABILITY:
                    if self.continuous:
                        self._pick_left_child_probability(k, action_value)
                    else:
                        self._pick_left_child_probability(
                            k, self.probabilities[action_value]
                        )
                elif action_type == ActionType.PICK_RIGHT_CHILD_PROBABILITY:
                    if self.continuous:
                        self._pick_right_child_probability(k, action_value)
                    else:
                        self._pick_right_child_probability(
                            k, self.probabilities[action_value]
                        )
                elif action_type == ActionType.PICK_TRIPLET:
                    self._pick_triplet(k)
                else:
                    raise NotImplementedError(
                        f"Unrecognized action type: {action_type}."
                    )

            return self.state, action, True
        else:
            self.done = True
            return self.state, action, True

    def step_backwards(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes a backward step given an action.

        Args
        ----
        action : tuple
            Action from the action space.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The state after executing the action.

        action : int
            Given action.

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # Replace the continuous value of threshold by -1 to allow checking it.
        action_to_check = self.action2representative(action)
        _, _, valid = super().step_backwards(
            action_to_check, skip_mask_check=skip_mask_check
        )
        return self.state, action, valid

    def set_state(self, state: List, done: Optional[bool] = False):
        """
        Sets the state and done. If done is True but incompatible with state (Stage is
        not COMPLETE), then force done False and print warning.
        """
        if done is True and self._get_stage() != Stage.COMPLETE:
            done = False
            warnings.warn(
                f"""
            Attempted to set state {self.state2readable(state)} with done = True, which
            is not compatible with the environment. Forcing done = False.
            """
            )
        return super().set_state(state, done)

    def sample_actions_batch_continuous(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
        states_from: Optional[List] = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs in the continuous mode.
        """
        n_states = policy_outputs.shape[0]
        logprobs = torch.zeros(n_states, device=self.device, dtype=self.float)

        # Handle discrete actions
        is_discrete = mask[:, self._action_index_pick_threshold]
        actions = [None] * n_states

        if torch.any(is_discrete):
            policy_outputs_discrete = policy_outputs[
                is_discrete, : self._index_continuous_policy_output
            ]
            actions_discrete, logprobs_discrete = super().sample_actions_batch(
                policy_outputs_discrete,
                mask[is_discrete, : self._index_continuous_policy_output],
                None,
                is_backward,
                sampling_method,
                temperature_logits,
                max_sampling_attempts,
            )
            logprobs[is_discrete] = logprobs_discrete
            actions_discrete = iter(actions_discrete)

        if not torch.all(is_discrete):
            # Handle continuous actions
            is_continuous = torch.logical_not(is_discrete)
            n_cont = is_continuous.sum()
            policy_outputs_cont = policy_outputs[
                is_continuous, self._index_continuous_policy_output :
            ]

            if sampling_method == "uniform":
                distr_threshold = torch.distributions.Uniform(
                    torch.zeros(n_cont, device=self.device),
                    torch.ones(n_cont, device=self.device),
                )
            elif sampling_method == "policy":
                mix_logits = policy_outputs_cont[:, 0::3]
                mix = Categorical(logits=mix_logits)
                alphas = policy_outputs_cont[:, 1::3]
                alphas = (
                    self.beta_params_max * torch.sigmoid(alphas) + self.beta_params_min
                )
                betas = policy_outputs_cont[:, 2::3]
                betas = (
                    self.beta_params_max * torch.sigmoid(betas) + self.beta_params_min
                )
                beta_distr = Beta(alphas, betas)
                distr_threshold = MixtureSameFamily(mix, beta_distr)

            thresholds = distr_threshold.sample()
            logprobs[is_continuous] = distr_threshold.log_prob(thresholds)

            actions_cont = [(ActionType.PICK_THRESHOLD, th.item()) for th in thresholds]
            actions_cont = iter(actions_cont)

        for i in range(n_states):
            if is_discrete[i]:
                actions[i] = next(actions_discrete)
            else:
                actions[i] = next(actions_cont)

        return actions, logprobs

    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
        states_from: Optional[List] = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.
        """
        if self.continuous:
            return self.sample_actions_batch_continuous(
                policy_outputs=policy_outputs,
                mask=mask,
                states_from=states_from,
                is_backward=is_backward,
                sampling_method=sampling_method,
                temperature_logits=temperature_logits,
                max_sampling_attempts=max_sampling_attempts,
            )
        else:
            return super().sample_actions_batch(
                policy_outputs=policy_outputs,
                mask=mask,
                states_from=states_from,
                is_backward=is_backward,
                sampling_method=sampling_method,
                temperature_logits=temperature_logits,
                max_sampling_attempts=max_sampling_attempts,
            )

    def get_logprobs_continuous(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", "n_dim"],
        mask: TensorType["n_states", "1"] = None,
        states_from: Optional[List] = None,
        is_backward: bool = False,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.
        """
        n_states = policy_outputs.shape[0]
        # TODO: make nicer
        if states_from is None:
            states_from = torch.empty(
                (n_states, self.policy_input_dim), device=self.device
            )
        logprobs = torch.zeros(n_states, device=self.device, dtype=self.float)
        # Discrete actions
        mask_discrete = mask[:, self._action_index_pick_threshold]
        if torch.any(mask_discrete):
            policy_outputs_discrete = policy_outputs[
                mask_discrete, : self._index_continuous_policy_output
            ]
            # states_from can be None because it will be ignored
            logprobs_discrete = super().get_logprobs(
                policy_outputs_discrete,
                actions[mask_discrete],
                mask[mask_discrete, : self._index_continuous_policy_output],
                None,
                is_backward,
            )
            logprobs[mask_discrete] = logprobs_discrete
        if torch.all(mask_discrete):
            return logprobs
        # Continuous actions
        mask_cont = torch.logical_not(mask_discrete)
        policy_outputs_cont = policy_outputs[
            mask_cont, self._index_continuous_policy_output :
        ]
        mix_logits = policy_outputs_cont[:, 0::3]
        mix = Categorical(logits=mix_logits)
        alphas = policy_outputs_cont[:, 1::3]
        alphas = self.beta_params_max * torch.sigmoid(alphas) + self.beta_params_min
        betas = policy_outputs_cont[:, 2::3]
        betas = self.beta_params_max * torch.sigmoid(betas) + self.beta_params_min
        beta_distr = Beta(alphas, betas)
        distr_threshold = MixtureSameFamily(mix, beta_distr)
        thresholds = actions[mask_cont, -1]
        logprobs[mask_cont] = distr_threshold.log_prob(thresholds)
        return logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", "n_dim"],
        mask: TensorType["n_states", "1"] = None,
        states_from: Optional[List] = None,
        is_backward: bool = False,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.
        """
        if self.continuous:
            return self.get_logprobs_continuous(
                policy_outputs,
                actions,
                mask,
                states_from,
                is_backward,
            )
        else:
            return super().get_logprobs(
                policy_outputs,
                actions,
                mask,
                states_from,
                is_backward,
            )

    def states2policy_mlp(
        self,
        states: Union[
            List[TensorType["state_dim"]], TensorType["batch_size", "state_dim"]
        ],
    ) -> TensorType["batch_size", "policy_input_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for an MLP policy model.
        It replaces the NaNs by -2s, removes the activity attribute, and explicitly
        appends the attribute vector of the active node (if present).
        """
        if isinstance(states, list):
            states = torch.stack(states)
        rows, cols = torch.where(states[:, :-1, Attribute.ACTIVE] == Status.ACTIVE)
        active_features = torch.full((states.shape[0], 1, 4), -2.0)
        active_features[rows] = states[rows, cols, : Attribute.ACTIVE].unsqueeze(1)
        states[states.isnan()] = -2
        states = torch.cat([states[:, :, : Attribute.ACTIVE], active_features], dim=1)
        return states.flatten(start_dim=1)

    @staticmethod
    @lru_cache(maxsize=10000)
    def get_leaf_paths(states: TensorType["batch_size", "state_dim"]) -> List:
        """
        Returns the path (all parent nodes) to all leaf nodes in a batch of states.
        """
        state_paths = []

        for state in states:
            paths = []
            leaves = Tree._find_leaves(state)

            if len(leaves) > 0:
                for l in leaves:
                    path = []
                    while l is not None:
                        path.append(l)
                        l = Tree._get_parent(l)
                    paths.append(sorted(path))
                state_paths.append(paths)
                paths = []
            else:
                state_paths.append([0])

        return state_paths

    def _get_leaf_paths(
        self, states: TensorType["batch_size", "state_dim"]
    ) -> TensorType["batch_size", "num_leaves_per_state", "policy_input_dim"]:
        """
        Returns the parent nodes with all the split decisions along the way to all leaf nodes
        for all batch states, replaces the NaN values in states by -2's.
        """
        # batch_size = states.shape[0]

        # # Replace NaN with -2 in one operation
        # states = torch.where(torch.isnan(states), torch.tensor(-2., device=self.device), states)

        # # Get leaf paths for all states
        # state_paths = self.get_leaf_paths(states)

        # # Create a tensor to hold all leaf paths
        # max_leaves = max(len(paths) for paths in state_paths)
        # leaf_paths = torch.full((batch_size, max_leaves, self.max_depth, 6), -2, device=self.device)

        # for i, paths in enumerate(state_paths):
        #     for j, path in enumerate(paths):
        #         for k, node in enumerate(path):
        #             sign = torch.tensor([node % 2], device=self.device)
        #             activity_features = states[i, node, -1][None,]
        #             leaf_paths[i, j, k] = torch.cat((states[i, node, :-1], sign, activity_features))

        # return leaf_paths.flatten(-2)

        state_paths = Tree.get_leaf_paths(states)
        max_depth = self.max_depth
        state_reps = None

        for idx, state in enumerate(states):
            state = state.to(self.device)
            state[state.isnan()] = -2
            leaves = Tree._find_leaves(state)
            leaf_paths = (
                torch.ones((2**max_depth, max_depth, 6), device=self.device) * -2
            )  # Might not be the best or most optimal setting

            if len(leaves) == 0:
                for j, p in enumerate(state_paths[idx]):
                    leaf_paths[j] = torch.cat(
                        (state[0], torch.tensor([-2], device=self.device))
                    )[
                        None, None
                    ]  # Append No Sign Token
            else:
                for j, p in enumerate(state_paths[idx]):
                    for i_, i in enumerate(p):
                        sign = torch.tensor([i % 2], device=self.device)
                        activity_features = state[i, -1][None,]
                        leaf_paths[j][i_] = torch.cat(
                            (state[i, :-1], sign, activity_features)
                        )

            if state_reps is None:
                state_reps = leaf_paths[None,]
            else:
                state_reps = torch.cat((state_reps, leaf_paths[None,]), 0)

        # TODO: Interpreting/Using the activity parameter more efficiently
        return state_reps.flatten(-2).to(self.device)

    def states2policy_transformer(
        self,
        states: Union[
            List[TensorType["state_dim"]], TensorType["batch_size", "state_dim"]
        ],
    ) -> TensorType["batch_size", "policy_input_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for a set transformer policy model.
        """
        if isinstance(states, list):
            states = torch.stack(states)
        return self._get_leaf_paths(states)

    def _attributes_to_readable(self, attributes: List) -> str:
        # Node type
        if attributes[Attribute.TYPE] == NodeType.CONDITION:
            node_type = "condition, "
        elif attributes[Attribute.TYPE] == NodeType.CLASSIFIER:
            node_type = "classifier, "
        else:
            return ""
        # Feature
        feature = f"feat. {str(attributes[Attribute.FEATURE])}, "
        # Decision threshold
        if attributes[Attribute.THRESHOLD] != -1:
            assert attributes[Attribute.TYPE] == 0
            threshold = f"th. {str(attributes[Attribute.THRESHOLD])}, "
        else:
            threshold = "th. -1, "
        # Class output
        if attributes[Attribute.PROBABILITY] != -1:
            probability_output = (
                f"probability {str(attributes[Attribute.PROBABILITY])}, "
            )
        else:
            probability_output = "probability -1, "
        if attributes[Attribute.ACTIVE] == Status.ACTIVE:
            active = " (active)"
        else:
            active = ""
        return node_type + feature + threshold + probability_output + active

    def state2readable(self, state=None):
        """
        Converts a state into human-readable representation.
        """
        if state is None:
            state = self.state.clone().detach()
        state = state.cpu().numpy()
        readable = ""
        for idx in range(self.n_nodes):
            attributes = self._attributes_to_readable(state[idx])
            if len(attributes) == 0:
                continue
            readable += f"{idx}: {attributes} | "
        readable += f"stage: {self._get_stage(state)}"
        return readable

    def _readable_to_attributes(self, readable: str) -> List:
        attributes_list = readable.split(", ")
        # Node type
        if attributes_list[0] == "condition":
            node_type = NodeType.CONDITION
        elif attributes_list[0] == "classifier":
            node_type = NodeType.CLASSIFIER
        else:
            node_type = -1
        # Feature
        feature = float(attributes_list[1].split("feat. ")[-1])
        # Decision threshold
        threshold = float(attributes_list[2].split("th. ")[-1])
        # Class output
        probability_output = float(attributes_list[3].split("probability ")[-1])
        # Active
        if "(active)" in readable:
            active = Status.ACTIVE
        else:
            active = Status.INACTIVE
        return [node_type, feature, threshold, probability_output, active]

    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.
        """
        readable_list = readable.split(" | ")
        state = torch.full((self.n_nodes + 1, Attribute.N), torch.nan)
        for el in readable_list[:-1]:
            node_index, attributes_str = el.split(": ")
            node_index = int(node_index)
            attributes = self._readable_to_attributes(attributes_str)
            for idx, att in enumerate(attributes):
                state[node_index, idx] = att
        stage = float(readable_list[-1].split("stage: ")[-1])
        state = self._set_stage(stage, state)
        return state

    @staticmethod
    def _batch_find_leaves(states: torch.Tensor) -> List[List[int]]:
        """
        Compute indices of leaves for a batch of states.
        """
        if states.ndim == 2:
            states = states.unsqueeze(0)

        leaves = torch.where(states[:, :-1, Attribute.TYPE] == NodeType.CLASSIFIER)
        return [leaves[1][leaves[0] == i].tolist() for i in range(states.shape[0])]

    @staticmethod
    def _find_leaves(state: torch.Tensor) -> List[int]:
        """
        Compute indices of leaves for a batch of sstates.
        """
        return torch.where(state[:-1, Attribute.TYPE] == NodeType.CLASSIFIER)[
            0
        ].tolist()

    @staticmethod
    def find_active(state: torch.Tensor) -> int:
        """
        Get index of the (only) active node. Assumes that active node exists
        (that we are in the middle of a macro step).
        """
        active = torch.where(state[:-1, Attribute.ACTIVE] == Status.ACTIVE)[0]
        assert len(active) == 1
        return active.item()

    @staticmethod
    def get_n_nodes(state: torch.Tensor) -> int:
        """
        Returns the number of nodes in a tree represented by the given state.
        """
        return (~torch.isnan(state[:-1, Attribute.TYPE])).sum()

    def get_policy_output_continuous(
        self, params: dict
    ) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled. It initializes the output tensor
        by using the parameters provided in the argument params.

        The output of the policy of a Tree environment consists of a discrete and
        continuous part. The discrete part (first part) corresponds to the discrete
        actions, while the continuous part (second part) corresponds to the single
        continuous action, that is the sampling of the threshold of a node classifier.

        The latter is modelled by a mixture of Beta distributions. Therefore, the
        continuous part of the policy output is vector of dimensionality c * 3,
        where c is the number of components in the mixture (self.components).
        The three parameters of each component are the following:

          1) the weight of the component in the mixture
          2) the logit(alpha) parameter of the Beta distribution to sample the
             threshold.
          3) the logit(beta) parameter of the Beta distribution to sample the
             threshold.

        Note: contrary to other environments where there is a need to model a mixture
        of discrete and continuous distributions (for example to consider the
        possibility of sampling the EOS action instead of a continuous action), there
        is no such need here because either the continuous action is the only valid
        action or it is not valid.
        """
        policy_output_discrete = torch.ones(
            self.action_space_dim, device=self.device, dtype=self.float
        )
        self._index_continuous_policy_output = len(policy_output_discrete)
        self._len_continuous_policy_output = self.components * 3
        policy_output_continuous = torch.ones(
            self._len_continuous_policy_output,
            device=self.device,
            dtype=self.float,
        )
        policy_output_continuous[1::3] = params["beta_alpha"]
        policy_output_continuous[2::3] = params["beta_beta"]
        return torch.cat([policy_output_discrete, policy_output_continuous])

    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled.
        """
        if self.continuous:
            return self.get_policy_output_continuous(params=params)
        else:
            return super().get_policy_output(params=params)

    def _get_prob_idx(self, prob: float) -> int:
        return self.prob2idx[np.round(prob, 2)]

    def _get_thresh_idx(self, threshold: float) -> int:
        return self.thresh2idx[np.round(threshold, 2)]

    # def get_mask_invalid_actions_forward(self, state=None, done=None):

    #     if state is None:
    #         state = self.state
    #     if done is None:
    #         done = self.done

    #     if done:
    #         return [True] * self.policy_output_dim

    #     stage = self._get_stage(state)

    #     state_np = state.cpu().numpy().astype(np.float32)
    #     stage_np = np.array(stage.item(), dtype=np.int64)
    #     thresholds_np = np.array(self.thresholds, dtype=np.float64)
    #     try:
    #         return get_mask_invalid_actions_forward_cy(
    #             state_np, stage_np, self.n_nodes, self.policy_output_dim,
    #             self._action_index_pick_leaf, self._action_index_eos,
    #             self._action_index_pick_leaf_probability, self._action_index_pick_feature,
    #             self._action_index_pick_threshold, self._action_index_pick_left_child_probability,
    #             self._action_index_pick_right_child_probability, self._action_index_pick_triplet,
    #             self.continuous, self.mask_redundant_choices, thresholds_np
    #         ).tolist()
    #     except Exception as e:
    #         print(e)
    #         code.interact(local=locals())

    def get_mask_invalid_actions_forward(
        self, state: Optional[torch.Tensor] = None, done: Optional[bool] = None
    ) -> List[bool]:
        if state is None:
            state = self.state
        if done is None:
            done = self.done

        if done:
            return [True] * self.policy_output_dim

        leaves = Tree._find_leaves(state)
        stage = self._get_stage(state)
        mask = [True] * self.policy_output_dim

        if stage == Stage.COMPLETE:
            # In the "complete" stage (in which there are no ongoing micro steps)
            # only valid actions are the ones for picking one of the leaves or EOS.
            for k in leaves:
                # Check if splitting the node wouldn't exceed max depth.
                if Tree._get_right_child(k) < self.n_nodes:
                    mask[self._action_index_pick_leaf + k] = False
            mask[self._action_index_eos] = False
        elif stage == Stage.LEAF:
            # Leaf was picked, only the dummy action of picking leaf probability is valid.
            k = self.find_active(state)
            if self.continuous:
                mask[self._action_index_pick_leaf_probability] = False
            else:
                prob_idx = self._get_prob_idx(state[k, Attribute.PROBABILITY].item())
                mask[self._action_index_pick_leaf_probability + prob_idx] = False
        elif stage == Stage.LEAF_PROBABILITY:
            # Leaf probability was picked, only picking the feature actions are valid.

            # Don't split on same feature twice along the same path if features are binary.
            if not self.continuous and (
                len(np.unique(self.X_train)) < 3 or len(self.thresholds) == 1
            ):
                k = self.find_active(state)
                parents = []
                while k is not None:
                    parents.append(k)
                    k = Tree._get_parent(k)

            for idx in range(
                self._action_index_pick_feature, self._action_index_pick_threshold
            ):
                mask[idx] = False

                # Don't split on same feature twice along the same path if features are binary.
                if not self.continuous and (
                    len(np.unique(self.X_train)) < 3 or len(self.thresholds) == 1
                ):
                    prev_feats = self.state[parents, Attribute.FEATURE]
                    if idx - self._action_index_pick_feature in prev_feats:
                        mask[idx] = True

        elif stage == Stage.FEATURE:
            # Feature was picked, now moving to pick a threshold.

            # Start by allowing all threshold actions.
            for idx in range(
                self._action_index_pick_threshold,
                self._action_index_pick_left_child_probability,
            ):
                mask[idx] = False  # Default to allowing all actions.

            if self.mask_redundant_choices:
                k = self.find_active(state)  # Current active node index
                feature_chosen = state[
                    k, Attribute.FEATURE
                ].item()  # The feature on which the current node will split
                path = []  # Store tuples of (threshold, direction based on parity)

                # Traverse from current node to root and collect thresholds with directions based on parity
                while k != 0:  # Assuming 0 is the root and has no parent
                    node_feature = state[k, Attribute.FEATURE].item()
                    if node_feature == feature_chosen:
                        node_threshold = state[k, Attribute.THRESHOLD].item()
                        direction = (
                            "left" if k % 2 == 1 else "right"
                        )  # Odd index -> left child, Even index -> right child
                        path.append((node_threshold, direction))
                    k = (
                        k - 1
                    ) // 2  # Get parent index in a binary tree stored in an array

                # Apply masking based on previously used thresholds along the path
                if path:
                    for idx in range(
                        self._action_index_pick_threshold + 1,
                        self._action_index_pick_left_child_probability - 1,
                    ):
                        current_threshold = self.thresholds[
                            idx - self._action_index_pick_threshold
                        ]
                        # Mask thresholds that contradict the path constraints
                        for threshold, direction in path:
                            if threshold < 0:  # Handling -1 fillers
                                continue
                            if (
                                (direction == "left" and current_threshold >= threshold)
                                or (
                                    direction == "right"
                                    and current_threshold <= threshold
                                )
                                or (current_threshold == threshold)
                            ):
                                mask[idx] = True
                                break

        elif stage == Stage.THRESHOLD:
            # Threshold was picked, only picking the left probability actions are valid.
            for idx in range(
                self._action_index_pick_left_child_probability,
                self._action_index_pick_right_child_probability,
            ):
                mask[idx] = False
        elif stage == Stage.LEFT_CHILD_PROBABILITY:
            # Left probability was picked, only picking the right probability actions are valid.
            for idx in range(
                self._action_index_pick_right_child_probability,
                self._action_index_pick_triplet,
            ):
                mask[idx] = False
        elif stage == Stage.RIGHT_CHILD_PROBABILITY:
            k = self.find_active(state)
            mask[self._action_index_pick_triplet + k] = False
        else:
            raise ValueError(f"Unrecognized stage {stage}.")

        return mask

    def get_mask_invalid_actions_backward_continuous(
        self,
        state: Optional[torch.Tensor] = None,
        done: Optional[bool] = None,
        parents_a: Optional[List] = None,
    ) -> List:
        """
        Simply appends to the standard "discrete part" of the mask a dummy part
        corresponding to the continuous part of the policy output so as to match the
        dimensionality.
        """
        return (
            super().get_mask_invalid_actions_backward(state, done, parents_a)
            + [True] * self._len_continuous_policy_output
        )

    def get_mask_invalid_actions_backward(
        self,
        state: Optional[torch.Tensor] = None,
        done: Optional[bool] = None,
        parents_a: Optional[List] = None,
    ) -> List:
        if self.continuous:
            return self.get_mask_invalid_actions_backward_continuous(
                state=state, done=done, parents_a=parents_a
            )
        else:
            return super().get_mask_invalid_actions_backward(
                state=state, done=done, parents_a=parents_a
            )

    def get_parents(
        self,
        state: Optional[torch.Tensor] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        if state is None:
            state = self.state
        if done is None:
            done = self.done

        if done:
            return [state], [self.eos]

        leaves = Tree._find_leaves(state)
        stage = self._get_stage(state)
        parents = []
        actions = []

        if stage == Stage.COMPLETE:
            # In the "complete" stage (in which there are no ongoing micro steps),
            # to find parents we first look for the nodes for which both children
            # are leaves, and then undo the last "pick triplet" micro step.
            leaves = set(leaves)
            triplets = []
            for k in leaves:
                if k % 2 == 1 and k + 1 in leaves:
                    triplets.append((Tree._get_parent(k), k, k + 1))
            for k_parent, k_left, k_right in triplets:
                parent = state.clone()
                attributes_parent = parent[k_parent]

                # Revert parent to the active status.
                attributes_parent[Attribute.ACTIVE] = Status.ACTIVE

                parent = self._set_stage(Stage.RIGHT_CHILD_PROBABILITY, parent)

                parents.append(parent)
                actions.append((ActionType.PICK_TRIPLET, k_parent))
        else:
            k = Tree.find_active(state)

            if stage == Stage.LEAF:
                # Revert self._pick_leaf.
                parent = state.clone()
                attributes = parent[k]

                parent = self._set_stage(Stage.COMPLETE, parent)
                attributes[Attribute.TYPE] = NodeType.CLASSIFIER
                attributes[Attribute.ACTIVE] = Status.INACTIVE

                parents.append(parent)
                actions.append((ActionType.PICK_LEAF, k))
            elif stage == Stage.LEAF_PROBABILITY:
                # Revert self._pick_leaf_probability.
                if self.continuous or k == 0:
                    outputs = [self.default_probability]
                else:
                    outputs = self.probabilities

                for output in outputs:
                    parent = state.clone()
                    attributes = parent[k]

                    parent = self._set_stage(Stage.LEAF, parent)
                    attributes[Attribute.PROBABILITY] = output

                    parents.append(parent)
                    if self.continuous:
                        actions.append((ActionType.PICK_LEAF_PROBABILITY, -1))
                    else:
                        actions.append(
                            (
                                ActionType.PICK_LEAF_PROBABILITY,
                                self._get_prob_idx(output),
                            )
                        )
            elif stage == Stage.FEATURE:
                # Revert self._pick_feature.
                parent = state.clone()
                attributes = parent[k]

                parent = self._set_stage(Stage.LEAF_PROBABILITY, parent)
                attributes[Attribute.FEATURE] = -1

                parents.append(parent)
                actions.append(
                    (ActionType.PICK_FEATURE, state[k][Attribute.FEATURE].item())
                )
            elif stage == Stage.THRESHOLD:
                # Revert self._pick_threshold.
                parent = state.clone()
                attributes = parent[k]

                parent = self._set_stage(Stage.FEATURE, parent)
                attributes[Attribute.THRESHOLD] = -1

                parents.append(parent)
                if self.continuous:
                    actions.append((ActionType.PICK_THRESHOLD, -1))
                else:
                    current_thresh = state[k][Attribute.THRESHOLD].item()
                    actions.append(
                        (
                            ActionType.PICK_THRESHOLD,
                            self._get_thresh_idx(current_thresh),
                        )
                    )
            elif stage == Stage.LEFT_CHILD_PROBABILITY:
                # Revert self._pick_left_child_probability.
                parent = state.clone()
                attributes_left = parent[Tree._get_left_child(k)]

                # Set action based on the probability of left child.
                current_prob = attributes_left[Attribute.PROBABILITY].item()
                if self.continuous:
                    action = (ActionType.PICK_LEFT_CHILD_PROBABILITY, current_prob)
                else:
                    action = (
                        ActionType.PICK_LEFT_CHILD_PROBABILITY,
                        self._get_prob_idx(current_prob),
                    )

                # Revert stage to "threshold".
                parent = self._set_stage(Stage.THRESHOLD, parent)

                # Reset left child attributes.
                attributes_left[:] = torch.nan

                parents.append(parent)
                actions.append(action)
            elif stage == Stage.RIGHT_CHILD_PROBABILITY:
                # Revert self._pick_right_child_probability.
                parent = state.clone()
                attributes_right = parent[Tree._get_right_child(k)]

                # Set action based on the probability of right child.
                current_prob = attributes_right[Attribute.PROBABILITY].item()
                if self.continuous:
                    action = (ActionType.PICK_RIGHT_CHILD_PROBABILITY, current_prob)
                else:
                    action = (
                        ActionType.PICK_RIGHT_CHILD_PROBABILITY,
                        self._get_prob_idx(current_prob),
                    )

                # Revert stage to "left probability".
                parent = self._set_stage(Stage.LEFT_CHILD_PROBABILITY, parent)

                # Reset right child attributes.
                attributes_right[:] = torch.nan

                parents.append(parent)
                actions.append(action)
            else:
                raise ValueError(f"Unrecognized stage {stage}.")

        return parents, actions

    @staticmethod
    def action2representative_continuous(action: Tuple) -> Tuple:
        """
        Replaces the continuous value of a PICK_THRESHOLD action by -1 so that it can
        be contrasted with the action space and masks.
        """
        if action[0] == ActionType.PICK_THRESHOLD:
            action = (ActionType.PICK_THRESHOLD, -1)
        return action

    def action2representative(self, action: Tuple) -> Tuple:
        if self.continuous:
            return self.action2representative_continuous(action=action)
        else:
            return super().action2representative(action=action)

    def get_max_traj_length(self) -> int:
        return self.n_nodes * Attribute.N

    @staticmethod
    def _get_graph(
        state: torch.Tensor,
        bidirectional: bool,
        *,
        graph: Optional[nx.DiGraph] = None,
        k: int = 0,
    ) -> nx.DiGraph:
        """
        Recursively convert state into a networkx directional graph.
        """
        if graph is None:
            graph = nx.DiGraph()

        attributes = state[k]
        graph.add_node(k, x=attributes, k=k)

        if attributes[Attribute.TYPE] != NodeType.CLASSIFIER:
            k_left = Tree._get_left_child(k)
            if not torch.any(torch.isnan(state[k_left])):
                Tree._get_graph(state, bidirectional, graph=graph, k=k_left)
                graph.add_edge(k, k_left)
                if bidirectional:
                    graph.add_edge(k_left, k)

            k_right = Tree._get_right_child(k)
            if not torch.any(torch.isnan(state[k_right])):
                Tree._get_graph(state, bidirectional, graph=graph, k=k_right)
                graph.add_edge(k, k_right)
                if bidirectional:
                    graph.add_edge(k_right, k)

        return graph

    def get_pyg_input_dim(self) -> int:
        return Tree.state2pyg(self.state, self.n_features).x.shape[1]

    @staticmethod
    def state2pyg(
        state: torch.Tensor,
        n_features: int,
        one_hot: bool = True,
        add_self_loop: bool = False,
    ) -> pyg.data.Data:
        """
        Convert given state into a PyG graph.
        """
        k = torch.nonzero(~state[:-1, Attribute.TYPE].isnan()).squeeze(-1)
        x = state[k].clone().detach()
        if one_hot:
            x = torch.cat(
                [
                    x[
                        :,
                        [
                            Attribute.TYPE,
                            Attribute.THRESHOLD,
                            Attribute.PROBABILITY,
                            Attribute.ACTIVE,
                        ],
                    ],
                    F.one_hot((x[:, Attribute.FEATURE] + 1).long(), n_features + 1),
                ],
                dim=1,
            )

        k_array = k.detach().cpu().numpy()
        k_mapping = {value: index for index, value in enumerate(k_array)}
        k_set = set(k_array)
        edges = []
        edge_attrs = []
        for k_i in k_array:
            if add_self_loop:
                edges.append([k_mapping[k_i], k_mapping[k_i]])
            if k_i > 0:
                k_parent = (k_i - 1) // 2
                if k_parent in k_set:
                    edges.append([k_mapping[k_parent], k_mapping[k_i]])
                    edge_attrs.append([1.0, 0.0])
                    edges.append([k_mapping[k_i], k_mapping[k_parent]])
                    edge_attrs.append([0.0, 1.0])
        if len(edges) == 0:
            edge_index = torch.empty((2, 0)).long()
            edge_attr = torch.empty((0, 2)).float()
        else:
            edge_index = torch.Tensor(edges).T.long()
            edge_attr = torch.Tensor(edge_attrs)

        return pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, k=k)

    def _state2pyg(self) -> pyg.data.Data:
        """
        Convert self.state into a PyG graph.
        """
        return Tree.state2pyg(self.state, self.n_features)

    @staticmethod
    def _load_dataset(data_path):
        data_path = Path(data_path)
        if data_path.suffix == ".csv":
            df = pd.read_csv(data_path)
            if df.columns[-1].lower() != "split":
                X_train = df.iloc[:, 0:-1].values
                y_train = df.iloc[:, -1].values
                X_test = None
                y_test = None
            else:
                if set(df.iloc[:, -1]) != {"train", "test"}:
                    raise ValueError(
                        f"Expected df['Split'] to have values in {{'train', 'test'}}, "
                        f"received {set(df.iloc[:, -1])}."
                    )
                X_train = df[df.iloc[:, -1] == "train"].iloc[:, 0:-2].values
                y_train = df[df.iloc[:, -1] == "train"].iloc[:, -2].values
                X_test = df[df.iloc[:, -1] == "test"].iloc[:, 0:-2].values
                y_test = df[df.iloc[:, -1] == "test"].iloc[:, -2].values
        elif data_path.suffix == ".pkl":
            with open(data_path, "rb") as f:
                dct = pickle.load(f)
                X_train = dct["X_train"]
                y_train = dct["y_train"]
                X_test = dct.get("X_test")
                y_test = dct.get("y_test")
        else:
            raise ValueError(
                "data_path must be a CSV (*.csv) or a pickled dict (*.pkl)."
            )
        return X_train, y_train, X_test, y_test

    @staticmethod
    def predict_proba(
        states: torch.Tensor,
        X: npt.NDArray,
        return_k: bool = False,
        k: int = 0,
        dirichlet: bool = False,
    ) -> Union[Union[float, np.ndarray], Tuple[Union[float, np.ndarray], int]]:

        states_np = states.cpu().numpy()
        if states_np.ndim == 2:
            states_np = states_np[None, :]

        if not states_np.flags["C_CONTIGUOUS"]:
            states_np = np.ascontiguousarray(states_np)

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        node_types = states_np[:, :, Attribute.TYPE].astype(np.int32)
        features = states_np[:, :, Attribute.FEATURE].astype(np.int32)
        thresholds = states_np[:, :, Attribute.THRESHOLD].astype(states_np.dtype)
        probabilities = states_np[:, :, Attribute.PROBABILITY].astype(states_np.dtype)
        if probabilities.ndim == 2:
            probabilities = probabilities[:, :, None]

        if X.ndim == 1:
            X = X.astype(states_np.dtype)[None,]
        else:
            X = X.astype(states_np.dtype)

        attribute_n = Attribute.N
        try:
            proba, node_index = batch_predict_proba_multiple_states_cython(
                states_np,
                X,
                node_types,
                features,
                thresholds,
                probabilities,
                attribute_n,
            )
        except Exception as e:
            print(e)
            code.interact(local=dict(globals(), **locals()))

        if not dirichlet:
            proba = proba  # No need to extract, as it's already a scalar

        if return_k:
            return proba, node_index
        else:
            return proba

    @staticmethod
    def predict(
        states: torch.Tensor,
        X: npt.NDArray,
        *,
        return_k: bool = False,
        k: int = 0,
        dirichlet: bool = False,
    ) -> Union[int, Tuple[int, int]]:

        output = Tree.predict_proba(
            states, X, return_k=return_k, k=k, dirichlet=dirichlet
        )

        if return_k:
            prob, k = output
        else:
            prob = output

        if dirichlet:
            if len(prob) == 0:
                pred = -1
            else:
                pred = np.argmax(prob, axis=-1)
        else:
            pred = int(np.round(prob))

        if return_k:
            return pred, k
        else:
            return pred

    def _predict_proba(
        self, X: npt.NDArray, *, return_k: bool = False
    ) -> Union[float, Tuple[float, int]]:
        return Tree.predict_proba(
            self.state, X, return_k=return_k, dirichlet=self.dirichlet
        )

    def _predict(
        self, X: npt.NDArray, *, return_k: bool = False
    ) -> Union[int, Tuple[int, int]]:
        return Tree.predict(self.state, X, return_k=return_k, dirichlet=self.dirichlet)

    @staticmethod
    def plot(state, path: Optional[Union[Path, str]] = None) -> None:
        """
        Plot current state of the tree.
        """
        graph = Tree._get_graph(state, bidirectional=False)

        labels = {}
        node_color = []
        for node in graph:
            x = graph.nodes[node]["x"]
            if x[Attribute.TYPE] == NodeType.CONDITION:
                labels[node] = (
                    rf"$x_{int(x[Attribute.FEATURE].item())}$ < "
                    rf"{np.round(x[Attribute.THRESHOLD].item(), 4)}"
                )
                node_color.append("white")
            else:
                labels[node] = (
                    f"p={('%.2f' % np.round(x[Attribute.PROBABILITY].item(), 2))[1:]}"
                )
                node_color.append("red")

        nx.draw(
            graph,
            graphviz_layout(graph, prog="dot"),
            labels=labels,
            node_color=node_color,
            with_labels=True,
            node_size=900,
            font_size=8,
        )
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
            plt.close()

    @staticmethod
    def _predict_samples(
        states: torch.Tensor, X: npt.NDArray, dirichlet: bool = False
    ) -> npt.NDArray:
        """
        Compute a matrix of predictions.

        Args
        ----
        states : Tensor
            Collection of sampled states with dimensionality (n_states, state_dim).

        X : NDArray
            Feature matrix with dimensionality (n_observations, n_features).

        Returns
        -------
        Prediction matrix with dimensionality (n_states, n_observations).

        """
        predictions = Tree.predict(states, X, dirichlet=dirichlet)
        return predictions

    @staticmethod
    def _compute_scores(
        predictions: npt.NDArray, y: npt.NDArray
    ) -> (dict, npt.NDArray):
        """
        Computes accuracy and balanced accuracy metrics for given predictions and ground
        truth labels.

        The metrics are computed in two modes: either as an average of scores calculated
        for individual trees (mean_tree_*), or as a single score calculated on a prediction
        made by the whole ensemble (forest_*), with ensembling done via prediction averaging.

        Args
        ----
        predictions: NDArray
            Prediction matrix with dimensionality (n_states, n_observations).

        y : NDArray
            Target vector with dimensionality (n_observations,).

        Returns
        -------
        Dictionary of (metric_name, score) key-value pairs.
        """
        scores = {}
        metrics = {"acc": accuracy_score, "bac": balanced_accuracy_score}

        for metric_name, metric_function in metrics.items():
            scores[f"mean_tree_{metric_name}"] = np.mean(
                [metric_function(y, y_pred) for y_pred in predictions]
            )
            scores[f"forest_{metric_name}"] = metric_function(
                y, predictions.mean(axis=0).round()
            )
        return scores

    @staticmethod
    def _plot_trees(
        states: List[torch.Tensor],
        scores: npt.NDArray,
        iteration: int,
    ):
        """
        Plots decision trees present in the given collection of states.

        Args
        ----
        states : Tensor
            Collection of sampled states with dimensionality (n_states, state_dim).

        scores : NDArray
            Collection of scores computed for the given states.

        iteration : int
            Current iteration (will be used to name the folder with trees).
        """
        path = Path(Path.cwd() / f"trees_{iteration}")
        path.mkdir()

        for i, (state, score) in enumerate(zip(states, scores)):
            Tree.plot(state, path / f"tree_{i}_{score:.4f}.png")

    def _sample_proba_dirichlet(self, states: torch.Tensor, prior=None, test=False):
        """
        Takes a batch of states and replaces their leaves' probabilities
        by sampling from Dirichlet distribution. This function is fully batched
        for both states and input data.
        """
        if prior is None:
            prior = self.prior

        # Choose the appropriate dataset
        X = self.X_test if test and self.X_test is not None else self.X_train
        y = self.y_test if test and self.X_test is not None else self.y_train

        # Batch find leaves for all states
        state_leaves = Tree._batch_find_leaves(states)

        # Perform batch prediction for all states and all data points
        _, nodes = Tree.predict(states, X, return_k=True, dirichlet=True)
        # nodes shape: [n_states, n_samples]

        # Initialize counters for all states
        max_leaves = max(len(leaves) for leaves in state_leaves)
        n_states = states.shape[0]
        leaf_success_counter = torch.zeros(
            (n_states, max_leaves, self.n_classes), dtype=int, device=states.device
        )
        leaf_total_flow = torch.zeros(
            (n_states, max_leaves), dtype=int, device=states.device
        )

        # Create a mapping from node indices to leaf indices for each state
        leaf_mapping = torch.full(
            (n_states, states.shape[1]), -1, dtype=int, device=states.device
        )
        for i, leaves in enumerate(state_leaves):
            leaf_mapping[i, leaves] = torch.arange(len(leaves), device=states.device)

        # Update counters based on predictions
        state_indices = torch.arange(n_states, device=states.device)[:, None].expand_as(
            torch.tensor(nodes)
        )
        leaf_indices = leaf_mapping[state_indices, nodes]
        valid_leaves = leaf_indices != -1

        leaf_total_flow.index_put_(
            (state_indices[valid_leaves], leaf_indices[valid_leaves]),
            torch.ones(1, dtype=int, device=states.device),
            accumulate=True,
        )

        y_expanded = torch.from_numpy(y)[None, :].expand(n_states, -1)
        leaf_success_counter.index_put_(
            (
                state_indices[valid_leaves],
                leaf_indices[valid_leaves],
                y_expanded[valid_leaves],
            ),
            torch.ones(1, dtype=int, device=states.device),
            accumulate=True,
        )

        # Create new states with space for class probabilities
        new_states = torch.cat(
            [
                states,
                torch.full(
                    (n_states, states.shape[1], self.n_classes),
                    torch.nan,
                    device=states.device,
                ),
            ],
            dim=2,
        )

        # Sample from Dirichlet distribution for each leaf in each state
        prior_tensor = torch.tensor(prior, device=states.device, dtype=torch.float32)
        for i, leaves in enumerate(state_leaves):
            n, k = (
                leaf_total_flow[i, : len(leaves)],
                leaf_success_counter[i, : len(leaves)],
            )
            dirichlet_params = k.float() + prior_tensor
            dirichlet = Dirichlet(dirichlet_params)
            samples = dirichlet.sample()
            new_states[i, leaves, Attribute.N :] = samples.to(new_states.dtype)

        return new_states

    def test(
        self,
        samples: Union[
            TensorType["n_trajectories", "..."], npt.NDArray[np.float32], List
        ],
    ) -> dict:
        """
        Computes a dictionary of metrics, as described in Tree._compute_scores, for
        both training and, if available, test data. If self.test_args['top_k_trees'] != 0,
        also plots top n trees and saves them in the log directory.

        Args
        ----
        samples : Tensor
            Collection of sampled states representing the ensemble.

        Returns
        -------
        Dictionary of (metric_name, score) key-value pairs.
        """
        result = {}

        if self.dirichlet:
            samples = torch.stack(samples, dim=0)
            train_samples = self._sample_proba_dirichlet(samples, test=False)

        result["mean_n_nodes"] = np.mean(
            [Tree.get_n_nodes(state) for state in train_samples]
        )
        train_predictions = Tree._predict_samples(
            train_samples, self.X_train, dirichlet=self.dirichlet
        )
        train_scores = Tree._compute_scores(train_predictions, self.y_train)
        for k, v in train_scores.items():
            result[f"train_{k}"] = v

        top_k_indices = None

        if self.test_args["top_k_trees"] != 0:
            if not hasattr(self, "test_iteration"):
                self.test_iteration = 0

            # Select top-k trees.
            accuracies = np.array(
                [accuracy_score(self.y_train, y_pred) for y_pred in train_predictions]
            )
            order = np.argsort(accuracies)[::-1]
            top_k_indices = order[: self.test_args["top_k_trees"]]

            # Plot trees.
            Tree._plot_trees(
                [train_samples[i][:, : -self.n_classes] for i in top_k_indices],
                accuracies[top_k_indices],
                self.test_iteration,
            )

            # Compute metrics for top-k trees.
            top_k_scores = Tree._compute_scores(
                train_predictions[top_k_indices], self.y_train
            )
            for k, v in top_k_scores.items():
                result[f"train_top_k_{k}"] = v

            # Compute metrics for the top-1 tree.
            top_1_index = top_k_indices[0]
            top_1_scores = Tree._compute_scores(
                np.array([train_predictions[top_1_index]]), self.y_train
            )
            for k, v in top_1_scores.items():
                result[f"train_top_1_{k}"] = v

            self.test_iteration += 1

        if self.X_test is not None:

            if self.dirichlet:
                test_samples = self._sample_proba_dirichlet(samples, test=True)
            test_predictions = Tree._predict_samples(
                test_samples, self.X_test, dirichlet=self.dirichlet
            )
            for k, v in Tree._compute_scores(test_predictions, self.y_test).items():
                result[f"test_{k}"] = v

            if top_k_indices is not None:
                # Compute metrics for top-k trees.
                top_k_scores = Tree._compute_scores(
                    test_predictions[top_k_indices], self.y_test
                )
                for k, v in top_k_scores.items():
                    result[f"test_top_k_{k}"] = v

                # Compute metrics for the top-1 tree.
                top_1_scores = Tree._compute_scores(
                    np.array([test_predictions[top_1_index]]), self.y_test
                )
                for k, v in top_1_scores.items():
                    result[f"test_top_1_{k}"] = v

        # result.update(self.bayesian_model_averaging(samples, test=(self.X_test is not None)))
        return result
