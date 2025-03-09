import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

class MCMCDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, mcmc_iterations=1000, alpha_value=0.1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.mcmc_iterations = mcmc_iterations
        self.tree = None
        self.alpha_value = alpha_value

    def _log_dirichlet(self, dirichlet_params):
        return np.sum(gammaln(dirichlet_params)) - gammaln(np.sum(dirichlet_params))

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

        log_prior = -(np.log2(4) + np.log2(X.shape[1]))*self.num_nodes()
         
        return log_likelihood + log_prior

    def _mcmc_split(self, X, y):
        n_features = X.shape[1]
        current_feature = np.random.randint(0, n_features)
        current_threshold = np.random.choice(X[:, current_feature])
        current_score = self._evaluate_split(X, y, current_feature, current_threshold)
        
        best_feature, best_threshold, best_score = current_feature, current_threshold, current_score
        
        for _ in range(self.mcmc_iterations):
            if np.random.random() < 0.5:
                proposed_feature = np.random.randint(0, n_features)
                proposed_threshold = np.random.choice(X[:, proposed_feature])
            else:
                proposed_feature = current_feature
                proposed_threshold = current_threshold + np.random.normal(0, 0.1 * (np.max(X[:, current_feature]) - np.min(X[:, current_feature])))
            
            proposed_score = self._evaluate_split(X, y, proposed_feature, proposed_threshold)
            
            if np.isfinite(proposed_score) and np.isfinite(current_score):
                acceptance_prob = min(1, np.exp(proposed_score - current_score))
            elif np.isfinite(proposed_score):
                acceptance_prob = 1
            else:
                acceptance_prob = 0
            
            if np.random.random() < acceptance_prob:
                current_feature, current_threshold, current_score = proposed_feature, proposed_threshold, proposed_score
                
                if current_score > best_score:
                    best_feature, best_threshold, best_score = current_feature, current_threshold, current_score
        
        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        if depth >= self.max_depth or n_samples < self.min_samples_split or len(unique_classes) == 1:
            counts = np.zeros(len(self.classes_))
            for i, c in enumerate(self.classes_):
                counts[i] = np.sum(y == c)
            return Node(value=counts)

        feature_index, threshold = self._mcmc_split(X, y)
        
        left_mask = X[:, feature_index] <= threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]
        
        left_subtree = self._grow_tree(X_left, y_left, depth + 1)
        right_subtree = self._grow_tree(X_right, y_right, depth + 1)

        return Node(feature_index=feature_index, threshold=threshold, left=left_subtree, right=right_subtree)

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        logger.info(f"Number of classes: {self.n_classes}")
        logger.info(f"Unique classes: {self.classes_}")
        logger.info(f"Input y shape: {y.shape}")
        logger.info(f"Input y unique values: {np.unique(y)}")
        self.tree = self._grow_tree(X, y)

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

def run_mcmc_dt_on_dataset(X_train, y_train, X_test, y_test):
    
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mcmc_dt = MCMCDecisionTree(max_depth=5)
        mcmc_dt.fit(X_train_scaled, y_train)

        y_pred = mcmc_dt.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy, mcmc_dt.num_nodes()
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return None, None
    
def experiment_on_datasets(seeds: List[int], n_trials: int=1, UCI: bool = False) -> Dict[str, Dict[str, float]]:

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
            # ("path/to/file.csv", "Dataset Name"),
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

        logger.info(f"Original y shape: {y.shape}")
        logger.info(f"Original y unique values: {np.unique(y)}")

        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)
            logger.info(f"After LabelEncoder - y unique values: {np.unique(y)}")
        
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Unique classes in dataset: {np.unique(y)}")
        
        accuracies = []
        n_nodes_list = []
        for seed in seeds:
            logger.info(f"\nRunning with seed {seed}")
            if UCI: 
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
            accs, sizes = [], []
            for _ in range(n_trials): 
                accuracy, n_nodes = run_mcmc_dt_on_dataset(X_train, y_train, X_test, y_test)
                if accuracy is not None and n_nodes is not None:
                    accs.append(accuracy)
                    sizes.append(n_nodes)
                accuracies.append(np.mean(accs))
                n_nodes_list.append(np.mean(sizes))
                logger.info(f"Seed {seed} - Accuracy: {np.mean(accs):.4f}, Nodes: {np.mean(sizes)}")
            if accuracy is not None and n_nodes is not None:
                accuracies.append(accuracy)
                n_nodes_list.append(n_nodes)
                logger.info(f"Seed {seed} - Accuracy: {accuracy:.4f}, Nodes: {n_nodes}")
        
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

# Run the experiments
np.random.seed(42)
seeds = np.array([1])  # Using multiple seeds for more robust results
results = experiment_on_datasets(seeds, n_trials=5)

# Print final results
logger.info("\nFinal Results:")
for dataset, metrics in results.items():
    logger.info(f"{dataset}:")
    logger.info(f"  Accuracy - Mean: {metrics['mean_accuracy']:.4f}, Std: {metrics['std_accuracy']:.4f}")
    logger.info(f"  Nodes    - Mean: {metrics['mean_nodes']:.2f}, Std: {metrics['std_nodes']:.2f}")