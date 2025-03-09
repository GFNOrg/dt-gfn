import os
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
from typing import List, Dict
from scipy.special import gammaln

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class SMCDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, n_particles=10, alpha_value=0.1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_particles = n_particles
        self.tree = None
        self.alpha_value = alpha_value

    def _log_dirichlet(self, dirichlet_params):
        return np.sum(gammaln(dirichlet_params)) - gammaln(np.sum(dirichlet_params)) 

    def _initialize_particles(self, n_features):
        return {
            'feature_index': np.random.randint(0, n_features, self.n_particles),
            'threshold': np.random.uniform(0, 1, self.n_particles)
        }

    def _move_particles(self, particles, n_features):
        move_mask = np.random.random(self.n_particles) < 0.5
        particles['feature_index'][move_mask] = np.random.randint(0, n_features, np.sum(move_mask))
        particles['threshold'][~move_mask] += np.random.normal(0, 0.1, np.sum(~move_mask))
        return particles

    def _evaluate_split(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        left_y, right_y = y[left_mask], y[~left_mask]
        
        if len(left_y) == 0 or len(right_y) == 0:
            return -np.inf
        
        left_counts = np.zeros(len(self.classes_))
        right_counts = np.zeros(len(self.classes_))
        
        for i, c in enumerate(self.classes_):
            left_counts[i] = np.sum(left_y == c)
            right_counts[i] = np.sum(right_y == c)
        
        alpha = np.ones(len(self.classes_)) * self.alpha_value
        
        log_likelihood = (
            self._log_dirichlet(left_counts + alpha) - self._log_dirichlet(alpha) +
            self._log_dirichlet(right_counts + alpha) - self._log_dirichlet(alpha)
        )
        
        log_prior = 0 #-(np.log2(4) + np.log2(X.shape[1]))*self.num_internal_nodes() 
        return log_likelihood + log_prior

    def _smc_split(self, X, y):
        particles = self._initialize_particles(X.shape[1])
        
        for _ in range(20):  # Number of SMC iterations
            particles = self._move_particles(particles, X.shape[1])
            
            weights = np.array([self._evaluate_split(X, y, p_f, p_t) 
                                for p_f, p_t in zip(particles['feature_index'], particles['threshold'])])
            weights = np.where(np.isfinite(weights), np.exp(weights), 0)
            
            # Handle potential zero sum of weights
            if np.sum(weights) == 0:
                weights = np.ones_like(weights) / len(weights)
            else:
                weights /= np.sum(weights)
            
            # Check for NaN values and replace with uniform probabilities if necessary
            if np.any(np.isnan(weights)):
                logger.warning("NaN weights encountered. Using uniform probabilities.")
                weights = np.ones_like(weights) / len(weights)
            
            indices = np.random.choice(self.n_particles, size=self.n_particles, p=weights)
            particles = {k: v[indices] for k, v in particles.items()}
        
        best_index = np.argmax(weights)
        return particles['feature_index'][best_index], particles['threshold'][best_index]

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        if depth >= self.max_depth or n_samples < self.min_samples_split or len(unique_classes) == 1:
            counts = np.zeros(len(self.classes_))
            for i, c in enumerate(self.classes_):
                counts[i] = np.sum(y == c)
            return Node(value=counts)

        feature_index, threshold = self._smc_split(X, y)
        
        left_mask = X[:, feature_index] <= threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]
        
        left_subtree = self._grow_tree(X_left, y_left, depth + 1)
        right_subtree = self._grow_tree(X_right, y_right, depth + 1)

        return Node(feature_index=feature_index, threshold=threshold, left=left_subtree, right=right_subtree)

    def _compute_log_posterior(self, node, alpha):
        if node is None:
            return 0
        
        if node.value is not None:
            # Leaf node
            counts = node.value
            log_likelihood = self._log_dirichlet(counts + alpha) - self._log_dirichlet(alpha)
            log_prior = 0 # -(np.log2(4) + np.log2(self.n_features)) * self.num_internal_nodes()
            return log_likelihood + log_prior
        
        left_log_post = self._compute_log_posterior(node.left, alpha)
        right_log_post = self._compute_log_posterior(node.right, alpha)
        
        return left_log_post + right_log_post

    def log_posterior(self):
        alpha = np.ones(len(self.classes_)) * self.alpha_value
        return self._compute_log_posterior(self.tree, alpha) 

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        logger.info(f"Number of classes: {self.n_classes}")
        logger.info(f"Unique classes: {self.classes_}")
        logger.info(f"Input y shape: {y.shape}")
        logger.info(f"Input y unique values: {np.unique(y)}")
        self.tree = self._grow_tree(X, y)
        log_posterior = self.log_posterior()
        logger.info(f"Log Posterior of the fitted tree: {log_posterior}")

    def _predict_sample(self, x, node):
        if node.value is not None:
            return self.classes_[np.argmax(node.value)]
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def num_nodes(self):
        return self._count_nodes(self.tree)

    def _count_nodes(self, node):
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def _count_internal_nodes(self, node):
        if node is None or (node.left is None and node.right is None):
            return 0
        return 1 + self._count_internal_nodes(node.left) + self._count_internal_nodes(node.right)

    def num_internal_nodes(self):
        return self._count_internal_nodes(self.tree)

def run_smc_dt_on_dataset(X_train, y_train, X_test, y_test, max_depth=5, n_particles=100, seed=42):
    try:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        smc_dt = SMCDecisionTree(max_depth=max_depth, n_particles=n_particles)
        smc_dt.fit(X_train_scaled, y_train)

        y_pred = smc_dt.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy, smc_dt.num_nodes()
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return None, None

def experiment_on_datasets(seeds: List[int], n_trials=10, UCI=True) -> Dict[str, Dict[str, float]]:
    if UCI:
        datasets = [
            (17, "BCW-D"),
            (109, "Wine"),
            (53, "Iris"),
            (850, "Raisin"),
        ]
    else:
        # Specify CSV files directly
        datasets = [
            # ("file_path.csv", "Dataset Name"),
        ]

    results = {}

    for dataset_info in datasets:
        if UCI:
            dataset_id, dataset_name = dataset_info
            logger.info(f"\nRunning experiment on {dataset_name} dataset...")
            dataset = fetch_ucirepo(id=dataset_id)
            X = dataset.data.features.values
            y = dataset.data.targets.values.ravel()
        else:
            file_path, dataset_name = dataset_info
            logger.info(f"\nRunning experiment on {dataset_name} dataset from {file_path}...")
            if not os.path.exists(file_path):
                logger.error(f"File {file_path} not found.")
                continue
            data = pd.read_csv(file_path)
            X = data.iloc[:, :-2].values  # Assuming last two columns are 'train'/'test' and the target
            y = data.iloc[:, -2].values
            split_column = data.iloc[:, -1].values  # 'train'/'test' column

            # Split based on the 'train'/'test' column
            X_train = X[split_column == 'train']
            y_train = y[split_column == 'train']
            X_test = X[split_column == 'test']
            y_test = y[split_column == 'test']

        accuracies = []
        n_nodes_list = []
        for seed in seeds:
            logger.info(f"\nRunning with seed {seed}")
            if UCI: 
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
            
            if y_train.dtype == object:
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)
            
            try:
                accs, sizes = [], []
                for _ in range(n_trials):
                    accuracy, n_nodes = run_smc_dt_on_dataset(X_train, y_train, X_test, y_test, seed=seed)
                    if accuracy is not None and n_nodes is not None:
                        accs.append(accuracy)
                        sizes.append(n_nodes)
                accuracies.append(np.mean(accs))
                n_nodes_list.append(np.mean(sizes))
                logger.info(f"Seed {seed} - Accuracy: {np.mean(accs):.4f}, Nodes: {np.mean(sizes)}")
            except Exception as e:
                logger.error(f"Error occurred with seed {seed}: {str(e)}")
        
        if accuracies and n_nodes_list:
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            mean_nodes = np.mean(n_nodes_list)
            std_nodes = np.std(n_nodes_list)
            
            results[dataset_name] = {
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "mean_nodes": mean_nodes,
                "std_nodes": std_nodes
            }
            
            logger.info(f"{dataset_name} - Mean Accuracy: {mean_accuracy:.4f}, Std: {std_accuracy:.4f}")
            logger.info(f"{dataset_name} - Mean Nodes: {mean_nodes:.2f}, Std: {std_nodes:.2f}")
        else:
            logger.error(f"Failed to process {dataset_name} dataset")
    
    return results

if __name__ == "__main__":
    # Run the experiments
    seeds = np.array([1])  # Using multiple seeds for more robust results
    results = experiment_on_datasets(seeds, n_trials=1, UCI=True)

    # Print final results
    logger.info("\nFinal Results:")
    for dataset, metrics in results.items():
        logger.info(f"{dataset}:")
        logger.info(f"  Accuracy - Mean: {metrics['mean_accuracy']:.4f}, Std: {metrics['std_accuracy']:.4f}")
        logger.info(f"  Nodes    - Mean: {metrics['mean_nodes']:.2f}, Std: {metrics['std_nodes']:.2f}")
