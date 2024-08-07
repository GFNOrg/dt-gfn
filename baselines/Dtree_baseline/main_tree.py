import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo
from dtree_source.dtree import DtreeDecisionTreeClassifier

def get_best_hyperparameters(X_train, y_train, X_val, y_val, max_depth=5, alphas=np.linspace(1, 10, 10), betas=np.linspace(1, 10, 10)):
    best_train_acc = -1
    best_params = (None, None)
    best_tree = None

    for a in alphas:
        for b in betas:
            tree = DtreeDecisionTreeClassifier(max_depth=max_depth, min_samples_split=10, a=a, b=b, alpha=0.01)
            tree.fit(X_train, y_train)
            y_train_pred = tree.predict(X_train)
            train_acc = np.mean(y_train_pred == y_train)

            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_params = (a, b)
                best_tree = tree

    return best_tree, best_params, best_train_acc

def run_dtree_on_dataset(X, y, max_depth=5, alphas=np.linspace(0, 10, 10), betas=np.linspace(0, 10, 10), seed=1):
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_tree, best_params, best_train_acc = get_best_hyperparameters(X_train_scaled, y_train, X_test_scaled, y_test, max_depth, alphas, betas)
    
    y_test_pred = best_tree.predict(X_test_scaled)
    test_acc = np.mean(y_test_pred == y_test)

    return best_params, best_train_acc, test_acc

def experiment_on_datasets_single_tree(seeds, max_depth=5, alphas=np.linspace(1, 10, 10), betas=np.linspace(1, 10, 10)):
    
    datasets = [
        # (267, "Banknote"),
        # (17, "BCW-D"),
        # (109, "Wine"),
        # (53, "Iris"),
        (850, "Raisin"),
        # (15, "BCW")
    ] 

    results = {}

    for dataset_id, dataset_name in datasets:
        print(f"Running experiment on {dataset_name} dataset...")
        dataset = fetch_ucirepo(id=dataset_id)
        X = dataset.data.features
        y = dataset.data.targets.values.ravel()

        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        per_seed_results = []
        for seed in seeds:
            best_params, best_train_acc, test_acc = run_dtree_on_dataset(X, y, max_depth=max_depth, alphas=alphas, betas=betas, seed=seed)
            per_seed_results.append((best_params, best_train_acc, test_acc))
            print(f"{dataset_name} {seed} - Best Params: {best_params}, Train Accuracy: {best_train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

        mean_test_accuracy = np.mean([result[2] for result in per_seed_results])
        std_test_accuracy = np.std([result[2] for result in per_seed_results])

        results[dataset_name] = {
            "mean_test_accuracy": mean_test_accuracy,
            "std_test_accuracy": std_test_accuracy
        }

        print(f"{dataset_name} - Mean Test Accuracy: {mean_test_accuracy:.4f}, Std: {std_test_accuracy:.4f}")
    return results

if __name__ == "__main__":
    
    seeds = range(1, 6)
    alphas = [0.5,0.65,0.8,0.95,1.1,1.25,1.4,1.55,1.7,1.85,2.0,2.15,2.3,2.45,2.6,2.75]
    betas = [i+1 for i in range(8)]
    results = experiment_on_datasets_single_tree(seeds)
    
    # Print final results
    print("\nFinal Results:")
    for dataset, metrics in results.items():
        print(f"{dataset} - Mean Test Accuracy: {metrics['mean_test_accuracy']:.4f}, Std: {metrics['std_test_accuracy']:.4f}")
