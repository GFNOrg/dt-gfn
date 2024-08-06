import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dpdt.utils.feature_selectors import CartAIGSelector
from dpdt import dpdt
from ucimlrepo import fetch_ucirepo
from dpdt.utils import build_mdp, CartAIGSelector, average_traj_length_in_mdp, extract_tree
from dpdt.solver import backward_induction_multiple_zetas
from dpdt.utils.mdp import State, build_mdp, eval_in_mdp

def run_dpdt_on_dataset(X, y, seed=42):
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

    # Train the DPDT model
    scores, depths, nodes, time_ = dpdt(
        X_train_scaled, y_train,
        aig_selector,
        zetas,
        max_depth=max_depth,
        verbose=False
    )

    # Evaluate the model on the test set
    best_zeta_index = np.argmax(scores)
    
    # Build the MDP for the best zeta
    tree = build_mdp(X_train_scaled, y_train, max_depth, aig_selector)
    policy = backward_induction_multiple_zetas(tree, zetas)
    
    # Evaluate on the test set
    init_obs = tree[0][0].obs
    test_accuracy = eval_in_mdp(X_test_scaled, y_test, policy, init_obs, best_zeta_index)

    return test_accuracy

def experiment_on_datasets(seeds):
    datasets = [
        # (267, "Banknote"),
        # (17, "BCW-D"),
        # (109, "Wine"),
        # (53, "Iris"),
        (850, "Raisin"),
        (15, "BCW")
    ]

    results = {}

    for dataset_id, dataset_name in datasets:
        print(f"Running experiment on {dataset_name} dataset...")
        dataset = fetch_ucirepo(id=dataset_id)
        X = dataset.data.features.values
        y = dataset.data.targets.values.ravel()

        # Check if y is categorical and encode if necessary
        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        accuracies = []
        for seed in seeds:
            accuracy = run_dpdt_on_dataset(X, y, seed)
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
results = experiment_on_datasets(seeds)

# Print final results
print("\nFinal Results:")
for dataset, metrics in results.items():
    print(f"{dataset} - Mean Accuracy: {metrics['mean_accuracy']:.4f}, Std: {metrics['std_accuracy']:.4f}")