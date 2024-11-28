"""
Runnable script with hydra capabilities
"""

import os
import pickle
import random
import sys
import code

import hydra
import pandas as pd

import cProfile 
import pstats

from pathlib import Path

from gflownet.utils.common import chdir_random_subdir
from gflownet.utils.policy import parse_policy_config


@hydra.main(config_path="./config", config_name="main", version_base="1.1")
def main(config):
    # TODO: fix race condition in a more elegant way
    chdir_random_subdir()

    # Get current directory and set it as root log dir for Logger
    cwd = os.getcwd()
    config.logger.logdir.root = cwd
    print(f"\nLogging directory of this run:  {cwd}\n")

    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)

    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
    # The proxy is required in the env for scoring: might be an oracle or a model
    proxy = hydra.utils.instantiate(
        config.proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    # The proxy is passed to env and used for computing rewards
    env = hydra.utils.instantiate(
        config.env,
        proxy=proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    # The policy is used to model the probability of a forward/backward action
    forward_config = parse_policy_config(config, kind="forward")
    backward_config = parse_policy_config(config, kind="backward")

    forward_policy = hydra.utils.instantiate(
        forward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
    )
    backward_policy = hydra.utils.instantiate(
        backward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
        base=forward_policy,
    )
    # State flow
    if config.gflownet.state_flow is not None:
        state_flow = hydra.utils.instantiate(
            config.gflownet.state_flow,
            env=env,
            device=config.device,
            float_precision=config.float_precision,
            base=forward_policy,
        )
    else:
        state_flow = None
    # GFlowNet Agent
    gflownet = hydra.utils.instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env=env,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        state_flow=state_flow,
        buffer=config.env.buffer,
        logger=logger,
    )

    # Train GFlowNet
    profiler = cProfile.Profile()
    profiler.enable()
    gflownet.train()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10) 

    # Sample from trained GFlowNet
    if config.n_samples > 0 and config.n_samples <= 1e5:
        batch, times = gflownet.sample_batch(n_forward=config.n_samples, train=False)
        x_sampled = batch.get_terminating_states(proxy=True)
        energies = env.proxy(x_sampled)
        x_sampled = batch.get_terminating_states()
        df = pd.DataFrame(
            {
                "readable": [env.state2readable(x) for x in x_sampled],
                "energies": energies.tolist(),
            }
        )
        df.to_csv("gfn_samples.csv")
        dct = {"x": x_sampled, "energy": energies}
        pickle.dump(dct, open("gfn_samples.pkl", "wb"))

    # Print replay buffer
    if len(gflownet.buffer.replay) > 0:
        print("\nReplay buffer:")
        print(gflownet.buffer.replay)

    # Close logger
    gflownet.logger.end()

    if config.env._target_ == 'gflownet.envs.tree_acc.Tree' and config.SAVE == True:
        from helper.get_tree_stats import calculate_tree_accuracies, bayesian_model_averaging
        mean_accuracy, std_accuracy, top_1_accuracy, mean_top_10_accuracy, \
        mean_nodes, top_1_state, top_1_nodes = calculate_tree_accuracies(max_depth=config.env.max_depth, 
                                                                n_thresholds=config.env.n_thresholds, 
                                                                dataset=config.env.dataset, 
                                                                seed=config.env.split_seed, 
                                                                n_samples=config.n_samples, 
                                                                datapath=config.env.data_path,
                                                                path=cwd)
        scores = bayesian_model_averaging(
            path=cwd,
            datapath=config.env.data_path,
            max_depth=config.env.max_depth,
            n_thresholds=config.env.n_thresholds,
            n_samples=config.n_samples
        )
        print(f'\nMean Accuracy: {mean_accuracy:.4f}')
        print(f'Standard Deviation of Accuracy: {std_accuracy:.4f}')
        print(f'Top-1 Accuracy: {top_1_accuracy:.4f}')
        print(f'Top-1 Tree Number of Nodes: {top_1_nodes:.4f}')
        print(f'Mean Top-10 Accuracy: {mean_top_10_accuracy:.4f}')
        print(f'Mean Number of Nodes: {mean_nodes:.4f}')
        print(f'\nBayesian Model Averaging Scores:')
        for key, value in scores.items():
            print(f'{key}: {value:.4f}')

        with open(f'{config.env.results_path}/{config.env.dataset}_{config.env.split_seed}.txt', 'w') as f:
            f.write(f'Mean Accuracy: {mean_accuracy:.4f}\n')
            f.write(f'Standard Deviation of Accuracy: {std_accuracy:.4f}\n')
            f.write(f'Top-1 Accuracy: {top_1_accuracy:.4f}\n')
            f.write(f'Top-1 Tree Number of Nodes: {top_1_nodes:.4f}\n')
            f.write(f'Mean Top-10 Accuracy: {mean_top_10_accuracy:.4f}\n')
            f.write(f'Mean Number of Nodes: {mean_nodes:.4f}\n')
            for key, value in scores.items():
                f.write(f'{key}: {value:.4f}\n')
            f.write(f'Result Path: {cwd}\n')

def set_seeds(seed):
    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
    sys.exit()
