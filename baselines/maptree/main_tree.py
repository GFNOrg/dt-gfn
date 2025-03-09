import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from ucimlrepo import fetch_ucirepo
import traceback
from tqdm import tqdm
import time

from experiments.globals import run_search, save_results

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
POSTERIOR = {
    'alpha': 0.95,
    'beta': 0.5,
    'rho': [2.5, 2.5],
}
FINAL_RUN_TIME_LIMIT = 300  # 300 seconds
RHO = (2.5, 2.5)
SEEDS = [0,1,2,3,4]
TEST_SIZE = 0.2

# Test different num_expansions values
NUM_EXPANSIONS_LIST = [10, 100, 1000, 10000]

SEARCHERS_AND_PARAMS_LISTS = [
    ("MAPTree", [
        {
            "num_expansions": n,
            "time_limit": FINAL_RUN_TIME_LIMIT,
            **POSTERIOR,
        } for n in NUM_EXPANSIONS_LIST
    ]),
]

UCI_DATASET_IDS = [
    (53, "Iris"),
    (109, "Wine"),
    (17, "BCW-D"),
    (850, "Raisin"),
]

def binarize_features_bucketing(X, n_bins=3, encode='onehot-dense', strategy='quantile'):
    """Convert continuous features to binary using bucketing."""
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    X_binned = discretizer.fit_transform(X)
    return X_binned.astype(int)

def binarize_labels(y):
    """Convert multi-class labels to binary (one-vs-rest)."""
    unique_classes = np.unique(y)
    if len(unique_classes) == 2:
        return [y == y.max()]
    elif len(unique_classes) > 2:
        binary_labels = [(y == class_label).astype(int) for class_label in unique_classes[:2]]
        return binary_labels
    else:
        raise ValueError("Dataset doesn't have enough classes for classification.")



def run(dataset_id: int, dataset_name: str):
    logger.info(f"Performance comparison on UCI dataset: {dataset_name}")
    logger.info("=====================================================")

    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features.values
    y = dataset.data.targets.values.ravel()

    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    y_binary_list = binarize_labels(y)
    
    results = []
    
    for seed in SEEDS:
        logger.info(f"\nRunning experiment with seed {seed}")
        
        seed_results = []
        for binary_index, y_binary in enumerate(y_binary_list):
            logger.info(f"Binary classification task {binary_index + 1}/{len(y_binary_list)}")
            start_time = time.time()
            X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=TEST_SIZE, random_state=seed, stratify=y_binary)
            
            X_train_binary = binarize_features_bucketing(X_train)
            X_test_binary = binarize_features_bucketing(X_test)

            for searcher, params_list in SEARCHERS_AND_PARAMS_LISTS:
                logger.info(f"Searcher: {searcher}")
                for j, params in enumerate(params_list):
                    logger.info(f"Params: {params}")

                    try:
                        
                        result = run_search(searcher, X_train_binary, y_train, **params)
                        
                        
                        if result is None:
                            logger.warning("Run Failed!!!")
                            continue

                        tree = result['tree']
                        search_time = result['time']
                        timeout = result['timeout']
                        lower_bound = result.get('lower_bound', None)
                        upper_bound = result.get('upper_bound', None)

                        train_acc = (tree.predict(X_train_binary) == y_train).mean()
                        test_acc = (tree.predict(X_test_binary) == y_test).mean()
                        train_sll = tree.log_likelihood(X_train_binary, y_train, rho=RHO) / len(y_train)
                        test_sll = tree.log_likelihood(X_test_binary, y_test, rho=RHO) / len(y_test)
                        size = tree.size()
                        end_time = time.time()
                        logger.info(f"Timed Out: {timeout}")
                        logger.info(f"Lower Bound: {lower_bound}")
                        logger.info(f"Upper Bound: {upper_bound}")
                        logger.info(f"Train Accuracy: {train_acc:.4f}")
                        logger.info(f"Test Accuracy: {test_acc:.4f}")
                        logger.info(f"Tree Size: {size}")
                        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

                        seed_results.append({
                            'seed': seed,
                            'searcher': searcher,
                            'params_id': j,
                            'num_expansions': params['num_expansions'],
                            'binary_task': binary_index,
                            'tree': str(tree),
                            'search_time': search_time,
                            'total_time': end_time - start_time,
                            'train_acc': train_acc,
                            'test_acc': test_acc,
                            'train_sll': train_sll,
                            'test_sll': test_sll,
                            'size': size,
                            'timeout': timeout,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                        })
                    except Exception as e:
                        logger.error(f"Error occurred: {str(e)}")
                        logger.error(traceback.format_exc())
        
        # Combine results from binary tasks
        for params in params_list:
            num_expansions = params['num_expansions']
            relevant_results = [r for r in seed_results if r['num_expansions'] == num_expansions]
            combined_result = {
                'seed': seed,
                'num_expansions': num_expansions,
                'test_acc': np.mean([r['test_acc'] for r in relevant_results]),
                'size': np.mean([r['size'] for r in relevant_results]),
                'total_time': np.sum([r['total_time'] for r in relevant_results]),
                'lower_bound': np.mean([r['lower_bound'] for r in relevant_results if r['lower_bound'] is not None]),
                'upper_bound': np.mean([r['upper_bound'] for r in relevant_results if r['upper_bound'] is not None]),
            }
            results.append(combined_result)

    save_results(pd.DataFrame(results), "uci", dataset_name)

    # Calculate overall statistics
    df = pd.DataFrame(results)
    metrics = df.groupby('num_expansions').agg({
        'test_acc': ['mean', 'std'],
        'size': ['mean', 'std'],
        'total_time': ['mean', 'std'],
        'lower_bound': ['mean', 'std'],
        'upper_bound': ['mean', 'std']
    }).reset_index()
    
    metrics.columns = ['num_expansions', 'mean_accuracy', 'std_accuracy', 'mean_nodes', 'std_nodes', 
                       'mean_time', 'std_time', 'mean_lower_bound', 'std_lower_bound', 
                       'mean_upper_bound', 'std_upper_bound']

    return metrics

if __name__ == "__main__":
    all_results = {}
    for dataset_id, dataset_name in tqdm(UCI_DATASET_IDS, desc="Datasets"):
        all_results[dataset_name] = run(dataset_id, dataset_name)
    
    logger.info("\nFinal Results for all datasets:")
    logger.info("===============================")
    for dataset_name, metrics in all_results.items():
        logger.info(f"\nDataset: {dataset_name}")
        
        # Overall statistics across all num_expansions
        overall_metrics = metrics.mean()
        logger.info(f"Overall Performance:")
        logger.info(f"  Accuracy - Mean: {overall_metrics['mean_accuracy']:.4f}, Std: {overall_metrics['std_accuracy']:.4f}")
        logger.info(f"  Nodes    - Mean: {overall_metrics['mean_nodes']:.2f}, Std: {overall_metrics['std_nodes']:.2f}")
        logger.info(f"  Time     - Mean: {overall_metrics['mean_time']:.4f}s, Std: {overall_metrics['std_time']:.4f}s")
        
        logger.info("\nPerformance by num_expansions:")
        for _, row in metrics.iterrows():
            logger.info(f"  Num Expansions: {row['num_expansions']}")
            logger.info(f"    Accuracy     - Mean: {row['mean_accuracy']:.4f}, Std: {row['std_accuracy']:.4f}")
            logger.info(f"    Nodes        - Mean: {row['mean_nodes']:.2f}, Std: {row['std_nodes']:.2f}")
            logger.info(f"    Time         - Mean: {row['mean_time']:.4f}s, Std: {row['std_time']:.4f}s")
            logger.info(f"    Lower Bound  - Mean: {row['mean_lower_bound']:.4f}, Std: {row['std_lower_bound']:.4f}")
            logger.info(f"    Upper Bound  - Mean: {row['mean_upper_bound']:.4f}, Std: {row['std_upper_bound']:.4f}")
        
        # Analysis and remarks
        best_accuracy = metrics.loc[metrics['mean_accuracy'].idxmax()]
        worst_accuracy = metrics.loc[metrics['mean_accuracy'].idxmin()]
        
        logger.info("\nAnalysis:")
        logger.info(f"  Best accuracy achieved with {best_accuracy['num_expansions']} expansions: {best_accuracy['mean_accuracy']:.4f}")
        logger.info(f"  Worst accuracy achieved with {worst_accuracy['num_expansions']} expansions: {worst_accuracy['mean_accuracy']:.4f}")