import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dpdt.utils.feature_selectors import CartAIGSelector
from dpdt import dpdt
from ucimlrepo import fetch_ucirepo
from dpdt.utils import build_mdp, CartAIGSelector, average_traj_length_in_mdp, extract_tree
from dpdt.solver import backward_induction_multiple_zetas
from dpdt.utils.mdp import State, build_mdp, eval_in_mdp

def run_dpdt_forest_on_dataset(X, y, n_trees=10, seed=42):
    np.random.seed(seed)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set up the DPDT parameters
    max_depth = 5
    zetas = np.linspace(0, 1, 11)
    aig_selector = CartAIGSelector(depth=max_depth)

    # Train the DPDT forest
    forest = []
    for _ in range(n_trees):
        # Bootstrap sampling
        indices = np.random.choice(len(X_train_scaled), len(X_train_scaled), replace=True)
        X_bootstrap = X_train_scaled[indices]
        y_bootstrap = y_train[indices]

        scores, depths, nodes, time_ = dpdt(
            X_bootstrap, y_bootstrap,
            aig_selector,
            zetas,
            max_depth=max_depth,
            verbose=False
        )

        # Build the MDP for the best zeta
        best_zeta_index = np.argmax(scores)
        tree = build_mdp(X_bootstrap, y_bootstrap, max_depth, aig_selector)
        policy = backward_induction_multiple_zetas(tree, zetas)
        forest.append((tree, policy, best_zeta_index))

    # Evaluate the forest on the test set
    def forest_predict(X):
        predictions = []
        for tree, policy, best_zeta_index in forest:
            root_state = tree[0][0]
            tree_predictions = []
            for x in X:
                state = root_state
                while not state.is_terminal:
                    action = max(policy[state].items(), key=lambda x: x[1][best_zeta_index])[0]
                    state = state.step(action)
                tree_predictions.append(state.label)
            predictions.append(tree_predictions)
        return np.round(np.mean(predictions, axis=0)).astype(int)

    y_pred = forest_predict(X_test_scaled)
    test_accuracy = np.mean(y_pred == y_test)

    return test_accuracy

def experiment_on_datasets(seeds, n_trees=10):
    datasets = [
        (850, "Raisin"),
        (15, "BCW")
    ]

    results = {}

    for dataset_id, dataset_name in datasets:
        print(f"Running experiment on {dataset_name} dataset...")
        dataset = fetch_ucirepo(id=dataset_id)
        X = dataset.data.features
        y = dataset.data.targets.values.ravel()

        # Check if y is categorical and encode if necessary
        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        accuracies = []
        for seed in seeds:
            accuracy = run_dpdt_forest_on_dataset(X, y, n_trees=n_trees, seed=seed)
            accuracies.append(accuracy)
        
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        results[dataset_name] = {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy
        }
        
        print(f"{dataset_name} - Mean Accuracy: {mean_accuracy:.4f}, Std: {std_accuracy:.4f}")
    
    return results

# Run the experiments
np.random.seed(42)
seeds = np.array([0])
n_trees = 10
results = experiment_on_datasets(seeds, n_trees)

# Print final results
print("\nFinal Results:")
for dataset, metrics in results.items():
    print(f"{dataset} - Mean Accuracy: {metrics['mean_accuracy']:.4f}, Std: {metrics['std_accuracy']:.4f}")