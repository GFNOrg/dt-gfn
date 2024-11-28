import code
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from gflownet.envs.tree_acc import Tree
from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat
from scipy.special import gammaln
from torchtyping import TensorType

from ..envs.tree_acc_cython import compute_log_likelihood_cython


class CategoricalTreeProxy(Proxy):
    def __init__(
        self,
        alpha: str = "Uniform",
        alpha_value: float = 1.0,
        beta: float = 1.0,
        use_prior: bool = False,
        gamma: float = 1.0,
        mini_batch: bool = False,
        batch_size: int = 512,
        log_likelihood_only: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_prior = use_prior
        self.gamma = gamma
        self.beta = beta
        self.sigma = nn.Parameter(torch.tensor(1.0))
        self.phi = nn.Parameter(torch.tensor(1.0))
        self.X = None
        self.y = None
        self.env = None
        self.alpha_type = alpha
        self.alpha_value = alpha_value
        self.max_depth = None
        self.mini_batch = mini_batch
        self.batch_size = batch_size
        self.log_likelihood_only = log_likelihood_only

    def setup(self, env: Optional[Tree] = None):
        self.X = env.X_train
        self.y = env.y_train
        self.env = env
        if env.X_test is not None:
            self.X_test = env.X_test
            self.y_test = env.y_test
        self.max_depth = env.max_depth
        self.n_classes = len(np.unique(self.y))

        # Prior over classification parameters
        if self.alpha_type == "Uniform":
            self.alpha = torch.ones(self.n_classes) * self.alpha_value
        elif self.alpha_type == "Label_Counts":
            class_counts = np.bincount(self.y) + 1
            self.alpha = torch.tensor(
                class_counts / class_counts.sum() * self.alpha_value
            )
        elif self.alpha_type == "Custom":
            if env.prior is not None:
                self.alpha = env.prior
            else:
                self.alpha = torch.ones(self.n_classes) * self.alpha_value
        else:
            raise ValueError("Unknown alpha initialization method")

    def set_sigma_phi(self, sigma, phi):
        self.sigma = sigma
        self.phi = phi

    def __call__(
        self, states: TensorType["batch", "state_dim"], test=False
    ) -> TensorType["batch"]:

        log_dirichlet = lambda dirichlet_params: sum(
            [gammaln(i) for i in dirichlet_params]
        ) - gammaln(sum(dirichlet_params))

        energies = []

        for state in states:

            state_np = state.cpu().numpy().astype(np.float32)

            if not self.mini_batch:
                if np.array_equal(np.unique(self.y_test), np.unique(self.y)):
                    if test:
                        samples = self.X_test
                        labels = self.y_test
                    else:
                        samples = self.X
                        labels = self.y
                else:
                    samples = self.X
                    labels = self.y

            else:
                if np.array_equal(np.unique(self.y_test), np.unique(self.y)):
                    if test:
                        sample_indices = np.random.choice(
                            self.X_test.shape[0], self.batch_size, replace=False
                        )
                        samples = self.X_test[sample_indices]
                        labels = self.y_test[sample_indices]
                    else:
                        sample_indices = np.random.choice(
                            self.X.shape[0], self.batch_size, replace=False
                        )
                        samples = self.X[sample_indices]
                        labels = self.y[sample_indices]
                else:
                    sample_indices = np.random.choice(
                        self.X.shape[0], self.batch_size, replace=False
                    )
                    samples = self.X[sample_indices]
                    labels = self.y[sample_indices]
            try:
                n_leaves, log_likelihood = compute_log_likelihood_cython(
                    state_np,
                    samples,
                    labels,
                    (
                        self.alpha.numpy()
                        if isinstance(self.alpha, torch.Tensor)
                        else self.alpha
                    ),
                    self.n_classes,
                )
            except Exception as e:
                print(f"Error in compute_log_likelihood_cython: {str(e)}")
                raise

            if self.use_prior:
                log_prior = 0
                # Compute the BCART prior
                for i in range(2**self.max_depth - 1):
                    depth = np.floor(np.log2(i + 1))
                    p_split = (self.sigma * (1 + depth) ** (-self.phi)).clone().detach()
                    is_internal = (
                        1 if Tree._get_left_child(i) < 2**self.max_depth - 1 else 0
                    )
                    if is_internal:
                        log_prior += torch.log(p_split)
                    else:
                        log_prior += torch.log(1 - p_split)
            else:
                n_internal_nodes = Tree.get_n_nodes(state) - n_leaves
                log_prior = -(np.log(4) + np.log(self.env.n_features)) * (
                    n_internal_nodes
                )

            if self.log_likelihood_only:
                log_prior = -1e-3

            energies.append((len(self.X) / len(samples)) * (log_likelihood + log_prior))

        return torch.tensor(energies, dtype=torch.float, device=self.device)


class TreeProxy(Proxy):
    """
    Simple decision tree proxy that uses empirical frequency of correct predictions for
    computing likelihood, and the number of nodes in the tree for computing the prior.
    """

    def __init__(self, use_prior: bool = True, beta: float = 1.0, **kwargs):
        """
        Parameters
        ----------
        use_prior : bool
            Whether to use -likelihood * prior for energy computation or just the
            -likelihood.
        beta : float
            Beta coefficient in `prior = np.exp(-self.beta * n_nodes)`. Note that
            this is temporary prior implementation that was used for debugging,
            in combination with reward_func="boltzmann" it doesn't make much sense.
        """
        super().__init__(**kwargs)

        self.use_prior = use_prior
        self.beta = beta
        self.X = None
        self.y = None

    def setup(self, env: Optional[Tree] = None):
        self.X = env.X_train
        self.y = env.y_train

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        energies = []

        for state in states:
            predictions = []
            for x in self.X:
                predictions.append(Tree.predict(state, x))
            likelihood = (np.array(predictions) == self.y).mean()

            if self.use_prior:
                n_nodes = Tree.get_n_nodes(state)
                prior = np.exp(-self.beta * n_nodes)
            else:
                prior = 1

            energies.append(-likelihood * prior)

        return tfloat(energies, float_type=self.float, device=self.device)
