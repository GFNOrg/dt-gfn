import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo
from dtree_source.dtree import DtreeDecisionTreeClassifier

def get_best_hyperparameters(X_train, y_train, max_depth=5, alphas=np.linspace(1, 10, 10), betas=np.linspace(1, 10, 10)):
    best_train_acc = -1
    best_params = (None, None)
    best_tree = None
    train_accuracies = np.zeros((len(alphas), len(betas)))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            tree = DtreeDecisionTreeClassifier(max_depth=max_depth, min_samples_split=10, a=a, b=b, alpha=0.01)
            tree.fit(X_train, y_train)
            y_train_pred = tree.predict(X_train)
            train_acc = np.mean(y_train_pred == y_train)
            train_accuracies[i, j] = train_acc

            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_params = (a, b)
                best_tree = tree

    return best_tree, best_params, best_train_acc, train_accuracies

def run_dtree_on_dataset(X, y, max_depth=5, alphas=np.linspace(1, 10, 10), betas=np.linspace(1, 10, 10), seed=1):
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_tree, best_params, best_train_acc, train_accuracies = get_best_hyperparameters(X_train_scaled, y_train, max_depth, alphas, betas)
    
    y_test_pred = best_tree.predict(X_test_scaled)
    test_acc = np.mean(y_test_pred == y_test)

    num_nodes = best_tree.num_nodes()

    return best_params, best_train_acc, test_acc, num_nodes, train_accuracies

def plot_heatmap(mean_train_accuracies, alphas, betas, dataset_name):
    mpl.rcParams.update(mpl.rcParamsDefault)

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')
    plt.rcParams['figure.figsize'] = [6, 2.2]
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 9.5
    plt.rcParams['axes.titlesize'] = 'small'
    plt.rcParams['axes.titlepad'] = 3
    plt.rcParams['xtick.labelsize'] = 'x-small'
    plt.rcParams['ytick.labelsize'] = plt.rcParams['xtick.labelsize']
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['legend.handlelength'] = 1.2
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.edgecolor'] = '#333'
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['patch.linewidth'] = 0.9
    plt.rcParams['hatch.linewidth'] = 0.9
    plt.rcParams['axes.linewidth'] = 0.6
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['xtick.minor.width'] = 0.6
    plt.rcParams['ytick.major.width'] = plt.rcParams['xtick.major.width']
    plt.rcParams['ytick.minor.width'] = plt.rcParams['xtick.minor.width']

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=(10, 8))
    plt.imshow(mean_train_accuracies, interpolation='nearest', cmap='Blues')
    plt.title(f'Training Accuracy for {dataset_name}')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\alpha$')
    plt.colorbar(label='Training Accuracy')
    plt.xticks(np.arange(len(betas)), labels=[f'{b:.2f}' for b in betas], rotation=45)
    plt.yticks(np.arange(len(alphas)), labels=[f'{a:.2f}' for a in alphas])
    plt.tight_layout()

    # Create directory if it does not exist
    if not os.path.exists('figures'):
        os.makedirs('figures')

    # Save the heatmap
    plt.savefig(f'figures/{dataset_name}_heatmap.png')
    plt.close()

def experiment_on_datasets_single_tree(seeds, max_depth=5, alphas=np.linspace(1, 10, 10), betas=np.linspace(1, 10, 10)):
    datasets = [
        # (267, "Banknote"),
        # (17, "BCW-D"),
        # (109, "Wine"),
        (53, "Iris"),
        # (850, "Raisin"),
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
        num_nodes_list = []
        all_train_accuracies = []

        for seed in seeds:
            best_params, best_train_acc, test_acc, num_nodes, train_accuracies = run_dtree_on_dataset(X, y, max_depth=max_depth, alphas=alphas, betas=betas, seed=seed)
            per_seed_results.append((best_params, best_train_acc, test_acc))
            num_nodes_list.append(num_nodes)
            all_train_accuracies.append(train_accuracies)
            print(f"{dataset_name} {seed} - Best Params: {best_params}, Train Accuracy: {best_train_acc:.4f}, Test Accuracy: {test_acc:.4f}, Num Nodes: {num_nodes}")
        
        # Calculate mean training accuracy across all seeds
        mean_train_accuracies = np.mean(all_train_accuracies, axis=0)
        plot_heatmap(mean_train_accuracies, alphas, betas, dataset_name)

        mean_test_accuracy = np.mean([result[2] for result in per_seed_results])
        std_test_accuracy = np.std([result[2] for result in per_seed_results])
        mean_num_nodes = np.mean(num_nodes_list)
        std_num_nodes = np.std(num_nodes_list)

        results[dataset_name] = {
            "mean_test_accuracy": mean_test_accuracy,
            "std_test_accuracy": std_test_accuracy,
            "mean_num_nodes": mean_num_nodes,
            "std_num_nodes": std_num_nodes
        }

        print(f"{dataset_name} - Mean Test Accuracy: {mean_test_accuracy:.4f}, Std: {std_test_accuracy:.4f}, Mean Num Nodes: {mean_num_nodes:.2f}, Std Num Nodes: {std_num_nodes:.2f}")
    return results

if __name__ == "__main__":
    seeds = range(1, 6)
    alphas = [0.5, 0.65, 0.8, 0.95, 1.1, 1.25, 1.4, 1.55, 1.7, 1.85, 2.0, 2.15, 2.3, 2.45, 2.6, 2.75]
    betas = [i + 1 for i in range(8)]
    results = experiment_on_datasets_single_tree(seeds, alphas=alphas, betas=betas)

    # Print final results
    print("\nFinal Results:")
    for dataset, metrics in results.items():
        print(f"{dataset} - Mean Test Accuracy: {metrics['mean_test_accuracy']:.4f}, Std: {metrics['std_test_accuracy']:.4f}, Mean Num Nodes: {metrics['mean_num_nodes']:.2f}, Std Num Nodes: {metrics['std_num_nodes']:.2f}")
