import numpy as np
from dtree_source.dtree import DtreeRandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo


def run_dtree_forest_on_dataset(X, y, n_estimators=10, seed=42):
    np.random.seed(seed)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print(f"Unique classes in training set: {np.unique(y_train)}")

    # Create and train the DtreeRandomForestClassifier
    forest = DtreeRandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_split=2,
        a=0.5,
        b=1.0,
        alpha=0.01,
        n_jobs=-1,  # Use all available cores
        max_features="sqrt",
        bootstrap=True,
    )

    try:
        forest.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"Error during fitting: {str(e)}")
        return None

    # Evaluate the forest on the test set
    try:
        y_pred = forest.predict(X_test_scaled)
        test_accuracy = np.mean(y_pred == y_test)
        print(f"Predictions shape: {y_pred.shape}")
        print(f"Unique predicted classes: {np.unique(y_pred)}")
        return test_accuracy
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None


def experiment_on_datasets(seeds, n_estimators=10):
    datasets = [(850, "Raisin"), (15, "BCW")]

    results = {}

    for dataset_id, dataset_name in datasets:
        print(f"\nRunning experiment on {dataset_name} dataset...")
        dataset = fetch_ucirepo(id=dataset_id)
        X = dataset.data.features
        y = dataset.data.targets.values.ravel()

        print(f"Dataset shape: {X.shape}")
        print(f"Unique classes: {np.unique(y)}")

        # Check if y is categorical and encode if necessary
        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)

        accuracies = []
        for seed in seeds:
            print(f"\nRunning with seed {seed}")
            accuracy = run_dtree_forest_on_dataset(
                X, y, n_estimators=n_estimators, seed=seed
            )
            if accuracy is not None:
                accuracies.append(accuracy)

        if accuracies:
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)

            results[dataset_name] = {
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
            }

            print(
                f"{dataset_name} - Mean Accuracy: {mean_accuracy:.4f}, Std: {std_accuracy:.4f}"
            )
        else:
            print(f"No valid accuracies for {dataset_name}")

    return results


# Run the experiments
if __name__ == "__main__":
    np.random.seed(42)
    seeds = np.array([1, 42, 123])  # Using multiple seeds for robustness
    n_estimators = 10  # Reduced number of trees for faster debugging

    results = experiment_on_datasets(seeds, n_estimators)

    # Print final results
    print("\nFinal Results:")
    for dataset, metrics in results.items():
        print(
            f"{dataset} - Mean Accuracy: {metrics['mean_accuracy']:.4f}, Std: {metrics['std_accuracy']:.4f}"
        )
