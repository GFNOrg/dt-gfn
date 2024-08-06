from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from typing import List, Dict, Union, Tuple
from joblib import Parallel, delayed

class DtreeDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, a: float = 1.0, b: float = 2.0, alpha: float = 0.01):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.a = a
        self.b = b
        self.alpha = alpha
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        dataset = np.column_stack((X, y))
        self.tree = self._build_tree(dataset)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict(self.tree, row) for row in X])

    def _build_tree(self, dataset: np.ndarray) -> Dict:
        root = self._get_split(dataset)
        self._split(root, 1)
        return root

    def _get_split(self, dataset: np.ndarray) -> Dict:
        class_values = np.unique(dataset[:, -1])
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(dataset.shape[1] - 1):
            for value in np.unique(dataset[:, index]):
                groups = self._test_split(index, value, dataset)
                split_score = self._ab_tsallis(groups, class_values)
                if split_score < b_score:
                    b_index, b_value, b_score, b_groups = index, value, split_score, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def _test_split(self, index: int, value: float, dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left = dataset[dataset[:, index] < value]
        right = dataset[dataset[:, index] >= value]
        return left, right

    def _ab_tsallis(self, groups: List[np.ndarray], classes: np.ndarray) -> float:
        n_instances = sum(len(group) for group in groups)
        abt = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = sum((np.sum(group[:, -1] == c) / size) ** self.a for c in classes)
            tsallis = 1.0 - score ** self.b if self.a > 1 else score ** self.b - 1.0
            abt += tsallis * (size / n_instances)
        return abt

    def _split(self, node: Dict, depth: int):
        left, right = node['groups']
        del(node['groups'])
        if not len(left) or not len(right):
            node['left'] = node['right'] = self._to_terminal(np.vstack((left, right)))
            return
        if depth >= self.max_depth:
            node['left'], node['right'] = self._to_terminal(left), self._to_terminal(right)
            return
        if len(left) <= self.min_samples_split:
            node['left'] = self._to_terminal(left)
        else:
            node['left'] = self._get_split(left)
            self._split(node['left'], depth + 1)
        if len(right) <= self.min_samples_split:
            node['right'] = self._to_terminal(right)
        else:
            node['right'] = self._get_split(right)
            self._split(node['right'], depth + 1)

    def _to_terminal(self, group: np.ndarray) -> float:
        if len(group) == 0:
            return 0  # Default class if the group is empty
        classes, counts = np.unique(group[:, -1], return_counts=True)
        return classes[np.argmax(counts)]

    def _predict(self, node: Dict, row: np.ndarray) -> float:
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict(node['right'], row)
            else:
                return node['right']

class DtreeRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators: int = 100, max_depth: int = 5, min_samples_split: int = 2, 
                 a: float = 1.0, b: float = 2.0, alpha: float = 0.01, n_jobs: int = -1, 
                 max_features: Union[str, int, float] = 'sqrt', bootstrap: bool = True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.a = a
        self.b = b
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features = int(np.sqrt(self.n_features))
            elif self.max_features == 'log2':
                self.max_features = int(np.log2(self.n_features))
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * self.n_features)
        
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._build_tree)(X, y) for _ in range(self.n_estimators)
        )
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray) -> DtreeDecisionTreeClassifier:
        tree = DtreeDecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            a=self.a,
            b=self.b,
            alpha=self.alpha
        )
        if self.bootstrap:
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
        else:
            X_bootstrap, y_bootstrap = X, y
        
        feature_indices = np.random.choice(self.n_features, self.max_features, replace=False)
        X_subset = X_bootstrap[:, feature_indices]
        
        tree.fit(X_subset, y_bootstrap)
        return tree, feature_indices

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_tree)(tree, feature_indices, X) for tree, feature_indices in self.trees
        )
        predictions = np.array(predictions).T
        return np.array([np.bincount(pred.astype(int)).argmax() for pred in predictions])

    def _predict_tree(self, tree: DtreeDecisionTreeClassifier, feature_indices: np.ndarray, X: np.ndarray) -> np.ndarray:
        return tree.predict(X[:, feature_indices])
