import numpy as np
import pandas as pd
from scipy.stats import invgamma, norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
from typing import List, Dict
import random
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BayesianCART:
    def __init__(self, max_depth=5, min_samples_split=2, n_mcmc=1000, a=1, v=3, lambda_=1, alpha=0.95, beta=2.0, max_time=300):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_mcmc = n_mcmc
        self.a = a
        self.v = v
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta
        self.tree = None
        self.max_time = max_time

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        initial_tree = pd.DataFrame({
            'node_id': [1], 'parent_node': [0], 'left_child': [0], 'right_child': [0],
            'terminal': [-1], 'split_var': [''], 'split_point': [0.0], 'depth': [0]
        })
        
        initial_data = self.apply_splits(initial_tree, X)
        initial_data['y'] = y
        initial_summary = self.summarize_nodes(initial_tree, X, y)
        initial_summary = self.calc_si_ti(initial_summary, self.a, np.mean(y))
        
        tree_list = {
            'tree': initial_tree,
            'data': initial_data,
            'node_summary': initial_summary
        }
        
        start_time = time.time()
        for i in range(self.n_mcmc):
            if time.time() - start_time > self.max_time:
                print(f"Reached maximum time limit after {i} iterations.")
                break
            
            if i % 100 == 0:
                print(f"Iteration {i}/{self.n_mcmc}")
            
            move_df = self.mh_proposal(tree_list['tree'])
            tree_proposal = self.modify_tree(move_df['operation'].values[0], tree_list['tree'], X, y, self.min_samples_split)
            
            if not tree_proposal['conditional']:
                continue
            
            proposed_tree = tree_proposal['tree']
            proposed_data = self.apply_splits(proposed_tree, X)
            proposed_data['y'] = y
            proposed_summary = self.summarize_nodes(proposed_tree, X, y)
            proposed_summary = self.calc_si_ti(proposed_summary, self.a, np.mean(y))
            
            tree_prop_list = {
                'tree': proposed_tree,
                'data': proposed_data,
                'node_summary': proposed_summary
            }
            
            tree_ratio = self.prior_tree(tree_list['tree'], proposed_tree, X, self.min_samples_split, self.n_features, move_df['operation'].values[0])
            
            c_lhood = self.lhood_y_xt(tree_list['node_summary'], self.v, self.lambda_, self.a)
            n_lhood = self.lhood_y_xt(tree_prop_list['node_summary'], self.v, self.lambda_, self.a)
            
            update_proposal = self.mh_update_proposal(tree_list['tree'], tree_prop_list['tree'],
                                                      move_df['move_prob'].values[0], move_df['rev_prob'].values[0],
                                                      move_df['operation'].values[0])
            
            tree_list = self.mh_step(tree_prop_list, tree_list, n_lhood, c_lhood,
                                     update_proposal['rev_prob'], update_proposal['move_prob'], tree_ratio)
        
        self.tree = tree_list['tree']
        print(f"Fitting completed in {time.time() - start_time:.2f} seconds.")

    def predict(self, X):
        if self.tree is None or self.tree.empty:
            raise ValueError("The tree has not been fit yet.")
        return np.array([self._predict_sample(x) for x in X])

    def _predict_sample(self, x):
        node = self.tree.iloc[0]
        while node['terminal'] != -1:
            if node['split_var'] == '' or pd.isna(node['split_point']):
                # If we encounter an invalid split, return a random class
                return np.random.choice(self.classes_)
            
            if x[node['split_var']] <= node['split_point']:
                child_id = node['left_child']
            else:
                child_id = node['right_child']
            
            child_node = self.tree[self.tree['node_id'] == child_id]
            if child_node.empty:
                # If we can't find the child node, return a random class
                return np.random.choice(self.classes_)
            node = child_node.iloc[0]
        
        # We've reached a terminal node
        return self.classes_[np.argmax(self.compute_leaf_value(node['node_id']))]

    def compute_leaf_value(self, node_id):
        # This is a placeholder. In a full implementation, you would compute the leaf value based on the data in the node.
        return np.random.dirichlet(np.ones(self.n_classes))

    def num_nodes(self):
        return len(self.tree) if self.tree is not None else 0

    def modify_tree(self, operation, tree, X, y, min_nobs=5, check_min_nobs=True):
        data = pd.concat([pd.DataFrame(X), pd.Series(y, name='y')], axis=1)
        
        if not check_min_nobs:
            proposed_tree = self.perform_operation(operation, tree, X, y)
            return {'tree': proposed_tree, 'conditional': True}
        else:
            min_nobs_condition = False
            max_iter = 1000
            iter = 0
            
            while not min_nobs_condition and iter < max_iter and check_min_nobs:
                proposed_tree = self.perform_operation(operation, tree, X, y)
                
                if proposed_tree is None or proposed_tree.equals(tree):
                    return {'tree': tree, 'conditional': False}
                
                min_nobs_condition = self.check_nobs_terminal_nodes(proposed_tree, data, min_nobs)
                iter += 1
            
            if iter >= max_iter:
                return {'tree': tree, 'conditional': False}
            else:
                return {'tree': proposed_tree, 'conditional': True}

    def perform_operation(self, operation, tree, X, y):
        if operation == 'grow':
            return self.grow(tree, X, y)
        elif operation == 'prune':
            return self.prune(tree)
        elif operation == 'change':
            return self.change(tree, X, y)
        elif operation == 'swap':
            return self.swap(tree)
        else:
            return None

    def grow(self, df, X, y):
        terminal_rows = df[df['terminal'] == -1]
        if terminal_rows.empty:
            return df
        
        split_node = int(np.random.choice(terminal_rows['node_id'].values))
        feature_index = np.random.randint(0, X.shape[1])
        threshold = np.random.choice(X[:, feature_index])
        
        max_node = df['node_id'].max()
        df.loc[df['node_id'] == split_node, ['left_child', 'right_child', 'terminal', 'split_var', 'split_point']] = [
            max_node + 1, max_node + 2, 1, feature_index, threshold
        ]
        
        depth = df.loc[df['node_id'] == split_node, 'depth'].values[0]
        new_nodes = pd.DataFrame({
            'node_id': [max_node + 1, max_node + 2],
            'parent_node': [split_node, split_node],
            'left_child': [0, 0],
            'right_child': [0, 0],
            'terminal': [-1, -1],
            'split_var': ['', ''],
            'split_point': [0.0, 0.0],
            'depth': [depth + 1, depth + 1]
        })
        df = pd.concat([df, new_nodes], ignore_index=True)
        
        return df

    def prune(self, df):
        if len(df) <= 1 or df[df['terminal'] == 1].empty:
            return df

        terminal_rows = df[df['terminal'] == -1]
        terminal_ids = terminal_rows['node_id'].values
        
        if len(terminal_ids) == 0:
            return df
        
        while True:
            terminal_node = int(np.random.choice(terminal_ids))
            parent_node = df.loc[df['node_id'] == terminal_node, 'parent_node'].values[0]
            left_child_id = df.loc[df['node_id'] == parent_node, 'left_child'].values[0]
            right_child_id = df.loc[df['node_id'] == parent_node, 'right_child'].values[0]
            left_terminal_cond = df.loc[df['node_id'] == left_child_id, 'terminal'].values[0]
            right_terminal_cond = df.loc[df['node_id'] == right_child_id, 'terminal'].values[0]
            if left_terminal_cond == -1 and right_terminal_cond == -1:
                break
            
            # If we can't find a valid node to prune, return the original tree
            if len(terminal_ids) == 1:
                return df
            
            terminal_ids = terminal_ids[terminal_ids != terminal_node]
        
        df = df[~df['node_id'].isin([left_child_id, right_child_id])]
        df.loc[df['node_id'] == parent_node, ['left_child', 'right_child', 'terminal', 'split_var', 'split_point']] = [0, 0, -1, '', 0.0]
        
        return df

    def change(self, df, X, y):
        internal_rows = df[df['terminal'] == 1]
        if internal_rows.empty:
            return df
        
        change_node = int(np.random.choice(internal_rows['node_id'].values))
        feature_index = np.random.randint(0, X.shape[1])
        threshold = np.random.choice(X[:, feature_index])
        
        df.loc[df['node_id'] == change_node, ['split_var', 'split_point']] = [feature_index, threshold]
        
        return df

    def swap(self, df):
        swap_rows = df[(df['parent_node'] != 0) & (df['terminal'] == 1)]
        if swap_rows.empty:
            return df
        
        swap_child_id = int(np.random.choice(swap_rows['node_id'].values))
        swap_parent_id = df.loc[df['node_id'] == swap_child_id, 'parent_node'].values[0]
        
        swap_rule = df.loc[df['node_id'] == swap_child_id, ['split_var', 'split_point']]
        parent_rule = df.loc[df['node_id'] == swap_parent_id, ['split_var', 'split_point']]
        
        if swap_rule.empty or parent_rule.empty:
            return df
        
        df.loc[df['node_id'] == swap_child_id, ['split_var', 'split_point']] = parent_rule.values[0]
        df.loc[df['node_id'] == swap_parent_id, ['split_var', 'split_point']] = swap_rule.values[0]
        
        return df
    def apply_splits(self, tree, data):
        df = tree.copy()
        result_data = pd.DataFrame(data).copy()
        result_data['terminal_node'] = 1

        if len(df) == 1:
            return result_data

        splitting_rows = df[df['split_var'] != '']

        for _, row in splitting_rows.iterrows():
            split_node_id = row['node_id']
            split_var = row['split_var']
            split_point = row['split_point']
            left_child = row['left_child']
            right_child = row['right_child']

            mask = result_data['terminal_node'] == split_node_id
            if split_var in result_data.columns:
                result_data.loc[mask & (result_data[split_var] <= split_point), 'terminal_node'] = left_child
                result_data.loc[mask & (result_data[split_var] > split_point), 'terminal_node'] = right_child
            else:
                result_data.loc[mask, 'terminal_node'] = left_child

        return result_data

    def summarize_nodes(self, tree, X, y):
        data = pd.concat([pd.DataFrame(X), pd.Series(y, name='y')], axis=1)
        data = self.apply_splits(tree, data)
        
        if len(data['terminal_node'].unique()) == 1:
            node_summary = pd.DataFrame({
                'terminal_node_id': [data['terminal_node'].iloc[0]],
                'node_size': [len(data)],
                'node_mean': [data['y'].mean()],
                'node_var': [data['y'].var() if len(data) > 1 else 0]
            })
        else:
            node_mean = data.groupby('terminal_node')['y'].mean()
            node_var = data.groupby('terminal_node')['y'].var()
            node_size = data['terminal_node'].value_counts()

            all_indices = sorted(set(node_mean.index) | set(node_var.index) | set(node_size.index))
            
            node_summary = pd.DataFrame({
                'terminal_node_id': all_indices,
                'node_size': node_size.reindex(all_indices, fill_value=0),
                'node_mean': node_mean.reindex(all_indices, fill_value=0),
                'node_var': node_var.reindex(all_indices, fill_value=0)
            })
        
        return node_summary

    def calc_si_ti(self, node_summary, a, mu_bar):
        node_summary['si'] = (node_summary['node_size'] - 1) * node_summary['node_var'].fillna(0)
        first_part = (node_summary['node_size'] * a) / (node_summary['node_size'] + a)
        second_part = (node_summary['node_mean'] - mu_bar) ** 2
        node_summary['ti'] = first_part * second_part
        return node_summary

    def mh_proposal(self, tree):
        if len(tree) == 1:
            move = pd.DataFrame({'operation': ['grow'], 'move_prob': [1.0], 'rev_prob': [1.0]})
        elif len(tree) == 3:
            move_list = ['grow', 'prune', 'change']
            move_prob = np.array([0.25, 0.25, 0.5])
            move_prob /= move_prob.sum()  # Ensure probabilities sum to 1
            rev_prob = move_prob.copy()
            m = np.random.choice(range(3), p=move_prob)
            move = pd.DataFrame({'operation': [move_list[m]], 'move_prob': [move_prob[m]], 'rev_prob': [rev_prob[m]]})
        else:
            move_list = ['grow', 'prune', 'change', 'swap']
            move_prob = np.array([0.25, 0.25, 0.4, 0.1])
            move_prob /= move_prob.sum()  # Ensure probabilities sum to 1
            rev_prob = move_prob.copy()
            m = np.random.choice(range(4), p=move_prob)
            move = pd.DataFrame({'operation': [move_list[m]], 'move_prob': [move_prob[m]], 'rev_prob': [rev_prob[m]]})
        return move

    def mh_update_proposal(self, current_tree, new_tree, move_pr, rev_pr, operation):
        if operation == 'grow':
            terminal_nodes = max((current_tree['terminal'] == -1).sum(), 1)
            birth_prob = 1 / terminal_nodes
            av_parents = new_tree[new_tree['terminal'] == -1].groupby('parent_node').size()
            av_parents = av_parents[av_parents == 2]
            death_prob = 1 / max(len(av_parents), 1)
            move_pr *= birth_prob
            rev_pr *= death_prob
        elif operation == 'prune':
            terminal_nodes = max((new_tree['terminal'] == -1).sum(), 1)
            birth_prob = 1 / terminal_nodes
            av_parents = current_tree[current_tree['terminal'] == -1].groupby('parent_node').size()
            av_parents = av_parents[av_parents == 2]
            death_prob = 1 / max(len(av_parents), 1)
            move_pr *= death_prob
            rev_pr *= birth_prob
        return {'move_prob': move_pr, 'rev_prob': rev_pr}

    def mh_step(self, n_tree, c_tree, n_lhood, c_lhood, n_proposal, c_proposal, prior_ratio):
        # Use log probabilities to avoid numerical underflow
        log_r = (np.log(n_lhood + 1e-300) + np.log(n_proposal + 1e-300) - 
                 np.log(c_proposal + 1e-300) - np.log(c_lhood + 1e-300) + 
                 np.log(prior_ratio + 1e-300))
        log_alpha = min(0, log_r)
        
        if np.log(max(np.random.random(), 1e-300)) <= log_alpha:
            return n_tree
        else:
            return c_tree

    def lhood_y_xt(self, node_summary, v, lambda_, a):
        b = len(node_summary)
        n = node_summary['node_size'].sum()
        
        sum_st = (node_summary['si'].fillna(0) + node_summary['ti'].fillna(0)).sum()
        
        # Use log computations to avoid overflow
        log_prod_na = np.sum(np.log(np.sqrt(node_summary['node_size'] + a) + 1e-300))
        log_num = (b / 2) * np.log(a + 1e-300)
        log_denom = log_prod_na + ((n + v) / 2) * np.log(sum_st + v * lambda_ + 1e-300)
        
        log_den = log_num - log_denom
        return np.exp(log_den)

    def prior_tree(self, current_tree, new_tree, data, min_node_size, n_pred, update):
        if update in ['grow', 'prune']:
            prior_ratio = self.tree_prior_grow_prune(current_tree, new_tree, data, min_node_size, n_pred)
        elif update == 'swap':
            prior_t = self.tree_prior_swap(current_tree, data, min_node_size, n_pred)
            prior_tprime = self.tree_prior_swap(new_tree, data, min_node_size, n_pred)
            prior_tprime = max(prior_tprime, 1e-300)
            prior_ratio = prior_tprime / (prior_t + 1e-300)
        else:
            prior_ratio = 1
        return prior_ratio

    def tree_prior_grow_prune(self, current_tree, new_tree, data, min_node_size, n_pred):
        if len(current_tree) < len(new_tree):
            tree_lg, tree_sm = new_tree, current_tree
        else:
            tree_sm, tree_lg = new_tree, current_tree
        
        child_nodes = set(tree_lg['node_id']) - set(tree_sm['node_id'])
        parent_node = tree_lg.loc[tree_lg['node_id'].isin(child_nodes), 'parent_node'].unique()[0]
        
        p_depth = tree_sm.loc[tree_sm['node_id'] == parent_node, 'depth'].values[0]
        p_internal = self.alpha * (1 + p_depth) ** (-self.beta)
        p_term = 1 - p_internal
        
        c_depth = p_depth + 1
        c_term = (1 - self.alpha * (1 + c_depth) ** (-self.beta)) ** 2
        
        grow_prior = (p_internal * c_term) / (p_term + 1e-300)
        
        tree_prior = grow_prior if len(new_tree) > len(current_tree) else 1 / (grow_prior + 1e-300)
        return tree_prior

    def tree_prior_swap(self, tree, data, min_node_size, n_pred):
        int_tree = tree[tree['terminal'] == 1]
        term_node = np.ones(len(data))
        p_rule = np.zeros(len(int_tree))
        
        for i, row in int_tree.iterrows():
            h = np.where(term_node == row['node_id'])[0]
            split_var = row['split_var']
            split_pt = row['split_point']
            left_child = row['left_child']
            right_child = row['right_child']
            
            term_node[h] = np.where(data.iloc[h][split_var] <= split_pt, left_child, right_child)
            
            temp_data = data.iloc[h]
            av_pred_vec = self.get_avail_predictors(temp_data, min_node_size, n_pred)
            
            pr_pred = 1 / max((av_pred_vec > 0).sum(), 1)
            pr_val = 1 / max(av_pred_vec[split_var], 1)
            
            p_rule[i] = pr_pred * pr_val
        
        tree_prior = np.prod(p_rule)
        return tree_prior

    def get_avail_predictors(self, data, min_node_size, n_pred):
        av_pred_vec = np.zeros(n_pred)
        for i in range(n_pred):
            if data.iloc[:, i].dtype == 'object':
                value_df = data.iloc[:, i].value_counts().reset_index()
                lower_bd = min_node_size
                upper_bd = value_df['count'].sum() - min_node_size
                av_cutpoints = value_df[(value_df['count'] >= lower_bd) & (value_df['count'] <= upper_bd)]
                av_pred_vec[i] = len(av_cutpoints)
            elif data.iloc[:, i].dtype.name == 'category':
                value_df = data.iloc[:, i].value_counts().reset_index()
                lower_bd = min_node_size
                upper_bd = value_df['count'].sum() - min_node_size
                av_cutpoints = value_df[(value_df['count'] >= lower_bd) & (value_df['count'] <= upper_bd)]
                av_pred_vec[i] = len(av_cutpoints)
            else:
                value_df = data.iloc[:, i].value_counts().reset_index()
                value_df['below'] = value_df['count'].cumsum()
                value_df['above'] = len(data) - value_df['below']
                av_cutpoints = value_df[(value_df['below'] >= min_node_size) & (value_df['above'] >= min_node_size)]
                av_pred_vec[i] = len(av_cutpoints)
        
        av_pred_vec = pd.Series(av_pred_vec, index=data.columns[:n_pred])
        return av_pred_vec

    def check_nobs_terminal_nodes(self, df, data, min_nobs):
        applied_tree = self.apply_splits(df, data)
        terminal_node_ids = df[df['terminal'] == -1]['node_id']
        
        node_sizes = applied_tree['terminal_node'].value_counts()
        
        return all(node_sizes.get(node_id, 0) >= min_nobs for node_id in terminal_node_ids)



def run_mcmc_dt_on_dataset(X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mcmc_dt = BayesianCART(max_depth=5, n_mcmc=10000, max_time=300)  # Set maximum time to 5 minutes
    mcmc_dt.fit(X_train_scaled, y_train)

    try:
        y_pred = mcmc_dt.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        accuracy = None

    return accuracy, mcmc_dt.num_nodes()

# Function to run experiments on datasets
def experiment_on_datasets(seeds: List[int], n_trials: int=1) -> Dict[str, Dict[str, float]]:
    datasets = [
        (17, "BCW-D"),
        (109, "Wine"),
        (53, "Iris"),
        (850, "Raisin"),
    ]

    results = {}

    for dataset_id, dataset_name in datasets:
        logging.info(f"\nRunning experiment on {dataset_name} dataset...")
        dataset = fetch_ucirepo(id=dataset_id)
        X = dataset.data.features.values
        y = dataset.data.targets.values.ravel()

        logging.info(f"Original y shape: {y.shape}")
        logging.info(f"Original y unique values: {np.unique(y)}")

        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)
            logging.info(f"After LabelEncoder - y unique values: {np.unique(y)}")
        
        logging.info(f"Dataset shape: {X.shape}")
        logging.info(f"Unique classes in dataset: {np.unique(y)}")
        
        accuracies = []
        n_nodes_list = []
        for seed in seeds:
            logging.info(f"\nRunning with seed {seed}")
            accs, sizes = [], []
            for trial in range(n_trials): 
                logging.info(f"Trial {trial + 1}/{n_trials}")
                accuracy, n_nodes = run_mcmc_dt_on_dataset(X, y, seed)
                if accuracy is not None and n_nodes is not None:
                    accs.append(accuracy)
                    sizes.append(n_nodes)
            accuracies.append(np.mean(accs))
            n_nodes_list.append(np.mean(sizes))
            logging.info(f"Seed {seed} - Accuracy: {np.mean(accs):.4f}, Nodes: {np.mean(sizes)}")
        
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
            
            logging.info(f"{dataset_name} - Mean Accuracy: {mean_accuracy:.4f}, Std: {std_accuracy:.4f}")
            logging.info(f"{dataset_name} - Mean Nodes: {mean_nodes:.2f}, Std: {std_nodes:.2f}")
        else:
            logging.warning(f"Failed to process {dataset_name} dataset")
    
    return results

# Main execution
if __name__ == "__main__":
    np.random.seed(42)
    seeds = np.array([1, 2, 3, 4, 5])  # Using multiple seeds for more robust results
    results = experiment_on_datasets(seeds, n_trials=10)

    # Print final results
    logging.info("\nFinal Results:")
    for dataset, metrics in results.items():
        logging.info(f"{dataset}:")
        logging.info(f"  Accuracy - Mean: {metrics['mean_accuracy']:.4f}, Std: {metrics['std_accuracy']:.4f}")
        logging.info(f"  Nodes    - Mean: {metrics['mean_nodes']:.2f}, Std: {metrics['std_nodes']:.2f}")