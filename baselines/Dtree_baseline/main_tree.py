import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo
from dtree_source.dtree import DtreeDecisionTreeClassifier


def run_dtree_on_dataset(X, y, seed=1):
    np.random.seed(seed)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the DtreeDecisionTreeClassifier
    tree = DtreeDecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        a=0.5,
        b=1.0,
        alpha=0.01
    )
    tree.fit(X_train_scaled, y_train)

    # Evaluate the tree on the test set
    y_pred = tree.predict(X_test_scaled)
    test_accuracy = np.mean(y_pred == y_test)

    return test_accuracy

def experiment_on_datasets_single_tree(seeds):
    datasets = [
        # (267, "Banknote"),
        # (17, "BCW-D"),
        # (109, "Wine"),
        (53, "Iris"),
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
            accuracy = run_dtree_on_dataset(X, y, seed=seed)
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
if __name__ == "__main__":
    seeds = np.array([1])  # Using multiple seeds for robustness

    results = experiment_on_datasets_single_tree(seeds)

    # Print final results
    print("\nFinal Results:")
    for dataset, metrics in results.items():
        print(f"{dataset} - Mean Accuracy: {metrics['mean_accuracy']:.4f}, Std: {metrics['std_accuracy']:.4f}")