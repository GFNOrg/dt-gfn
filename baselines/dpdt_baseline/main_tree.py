import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from dpdt_source.dpdt.utils.feature_selectors import CartAIGSelector
from dpdt_source.dpdt import dpdt
from ucimlrepo import fetch_ucirepo
from dpdt_source.dpdt.utils import build_mdp, CartAIGSelector, average_traj_length_in_mdp, extract_tree
from dpdt_source.dpdt.solver import backward_induction_multiple_zetas
from dpdt_source.dpdt.utils.mdp import State, build_mdp, eval_in_mdp

def run_dpdt_on_dataset(X, y, seed=42):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Standardize the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set up the DPDT parameters
    max_depth = 5
    zetas = np.linspace(0, 1, 101)[1:-1]
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
    test_accuracy, model_size = eval_in_mdp(X_test_scaled, y_test, policy, init_obs, best_zeta_index)

    return test_accuracy, model_size

def experiment_on_datasets(seeds):
    datasets = [
        # (267, "Banknote"),
        (17, "BCW-D"),
        # (109, "Wine"),
        # (53, "Iris"),
        (850, "Raisin"),
        # (15, "BCW")
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
        
        accuracies, model_sizes = [], []
        for seed in seeds:
            accuracy, model_size = run_dpdt_on_dataset(X, y, seed)
            accuracies.append(accuracy)
            model_sizes.append(model_size)
        
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_model_size = np.mean(model_sizes)
        std_model_size = np.std(model_sizes)
        
        results[dataset_name] = {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy, 
            "mean_model_size": mean_model_size,
            "std_model_size": std_model_size
        }
        
        print(f"{dataset_name} - Mean Accuracy: {mean_accuracy:.4f}, Std: {std_accuracy:.4f}")
        print(f"{dataset_name} - Mean Model Size: {mean_model_size:.4f}, Std: {std_model_size:.4f}")
    
    return results

# Run the experiments
# np.random.seed(42)
seeds = np.array([1, 2, 3, 4, 5])
results = experiment_on_datasets(seeds)

# Print final results
print("\nFinal Results:")
for dataset, metrics in results.items():
    print(f"{dataset} - Mean Accuracy: {metrics['mean_accuracy']:.4f}, Std: {metrics['std_accuracy']:.4f}")
    print(f"{dataset} - Mean Model Size: {metrics['mean_model_size']:.4f}, Std: {metrics['std_model_size']:.4f}")