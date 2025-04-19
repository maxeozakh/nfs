import numpy as np
import pandas as pd
import torch
import zipfile
# import kaggle
import os
import math
import queue
import copy

from pathlib import Path

path = Path('titanic')
extract_path = Path('data/titanic')

# Download dataset to use locally
# if not path.exists():
#     kaggle.api.competition_download_cli(str(path))
#     zipfile.ZipFile(f'{path}.zip').extractall(extract_path)

assert os.path.isdir(
    extract_path) == True, f"Directory '{path}' should be created at this point"
assert len(os.listdir(extract_path)
           ) > 0, f"Directory '{path}' is empty somehow, we expect files here"


df = pd.read_csv(f'{extract_path}/train.csv')

# Handle missing values in the dataset
modes = df.mode().iloc[0]
df.fillna(modes, inplace=True)
# df.isna().sum()


# features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# For now we will work just with few cols, later we could try to extend
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
# sibsp	-- # of siblings / spouses aboard the Titanic
# parch -- # of parents / children aboard the Titanic
# embarked -- Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton


df['Sex'] = df['Sex'].map({'male': 1, 'female': 0}).astype('Int8')

train_index = int(np.floor(0.8 * len(df)))

X_tr = df[features][:train_index]
Y_tr = df['Survived'][:train_index].values
X_test = df[features][train_index:]
Y_test = df['Survived'][train_index:].values


i_to_column = {i: c for i, c in enumerate(X_tr.columns)}

# convert df to tenrsors
if isinstance(X_tr, pd.DataFrame):
    X_tr = torch.tensor(X_tr.astype(np.float32).values, dtype=torch.float32)
    Y_tr = torch.tensor(Y_tr, dtype=torch.float32)
    X_test = torch.tensor(X_test.astype(
        np.float32).values, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)


def oob_score(tree, x, y):
    mis_label = 0
    for i in range(len(x)):
        pred = predict_tree(tree, x[i])
        if pred != y[i]:
            mis_label += 1
    return mis_label / len(x)


def entropy(p):
    '''
    How uncertain our split is. If values are 50/50 -- very uncertain, 1.
    If nearly all values are the same (99/1) -- certainty is high, very low entropy value
    '''
    if p == 0 or p == 1:
        return 0

    else:
        return - (p * np.log2(p) + (1 - p) * np.log2(1-p))


assert entropy(.5) == 1
assert entropy(.99) < 0.1


def information_gain(left_child, right_child):
    '''
    Calculates how much uncertainty is reduced by doing the split. Higher gain = better split.
    Assumes binary classification with labels 0 and 1.
    '''

    # Combine both child sets to get the parent
    parent = torch.cat((left_child, right_child), dim=0)

    def proportion_ones(t):
        if t.numel() == 0:
            return 0
        return torch.sum(t == 1).item() / t.numel()

    # Proportions of class 1
    p_parent = proportion_ones(parent)
    p_left = proportion_ones(left_child)
    p_right = proportion_ones(right_child)

    # Entropies
    IG_p = entropy(p_parent)
    IG_l = entropy(p_left)
    IG_r = entropy(p_right)

    # Weights
    l_weight = left_child.numel() / parent.numel()
    r_weight = right_child.numel() / parent.numel()

    return IG_p - l_weight * IG_l - r_weight * IG_r


class DecisionTree:
    def __init__(self, x=X_tr, y=Y_tr, max_depth=5, min_samples_split=50):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        # ===== subset of features for each tree =====
        fts_len = len(features)
        fts_indices = range(0, fts_len)
        n_fts = round(fts_len * .3)
        self.fts = np.random.choice(fts_indices, n_fts, replace=False)

        # ===== bagging =====
        # TODO: potential easier way to init this stuff but i'm not sure rn
        # n = len(x)
        # indices = np.random.choice(n, size=n, replace=True)
        # # out-of-bag, for validation
        # oob_indices = [i for i in range(len(x)) if i not in indices]
        # return x[indices], y[indices], x[oob_indices], y[oob_indices]

        x_len = len(x)
        bootstrap_indices = list(
            (np.random.choice(range(x_len), x_len, replace=True)))
        self.x = x[bootstrap_indices]
        self.y = y[bootstrap_indices]

        self.counter = 0
        self.splits = []
        self.nodes = []
        self.branches = queue.Queue()
        self.node_id = 0
        self.tree = None

        self.branches.put([x[bootstrap_indices], 0])

        self.global_ig = -999

        tree = self.create_node(self.x, self.y)
        self.tree = tree

    def find_best_split(self, feature_index, x, y):
        best_split = None
        ig_max = -9999

        possible_splits = x[:, feature_index].unique()

        for split in possible_splits:
            left_mask = x[:, feature_index] <= split
            right_mask = x[:, feature_index] > split

            left = y[left_mask]
            right = y[right_mask]

            ig = information_gain(left, right)

            if ig > ig_max:
                ig_max = ig
                best_split = split

        if (ig_max < 0):
            raise ValueError('That means broken node in future')

        res = [feature_index, ig_max, best_split]
        self.splits.append(res)

    def create_node(self, x, y):
        assert self.counter < 1999, f"Woooooooo {len(self.nodes)}"
        self.counter += 1

        node = {
            'ft_idx': None,
            'x': x,
            'y': y,
            'split': None,
            'right_node': None,
            'left_node': None,
            'ig': -999,
        }

        for ft_index in self.fts:
            self.find_best_split(ft_index, x, y)

        for split in self.splits:
            ft_idx, ig, split_value = split
            if (ig > node['ig']):
                node['ft_idx'] = ft_idx
                node['ft'] = features[ft_idx]
                node['split'] = split_value
                node['ig'] = ig

        split_value = node['split']
        ft_idx = node['ft_idx']
        left_mask_x = x[:, ft_idx] <= split_value
        right_mask_x = x[:, ft_idx] > split_value

        x_left = x[left_mask_x]
        x_right = x[right_mask_x]

        y_left = y[left_mask_x]
        y_right = y[right_mask_x]

        self.splits = []

        if x_left.shape[0] > self.min_samples_split and x_right.shape[0] > self.min_samples_split:
            node['left_node'] = self.create_node(x_left, y_left)
            node['right_node'] = self.create_node(x_right, y_right)

        return node

    def predict(self, data, node=None, depth=0):
        if node is None:
            node = self.tree

        ft_idx = node['ft_idx']
        split_value = node['split']
        ft = node['ft']
        left_node = node['left_node']
        right_node = node['right_node']
        ig = node['ig']
        x = node['x']
        y = node['y']

        if left_node is None and right_node is None:
            return y.mean()

        if data[ft_idx] <= split_value:
            return self.predict(data, left_node, depth=depth+1)
        else:
            return self.predict(data, right_node, depth=depth+1)


def forest(n_trees=90):
    x = X_test
    y = Y_test
    data = x
    g_preds = []

    trees = [DecisionTree() for _ in range(0, n_trees)]

    for j, d in enumerate(data):
        preds = [trees[i].predict(d) for i in range(0, n_trees)]
        g_preds.append((torch.tensor(preds).mean() > 0.5).item()
                       == bool(y[j].item()))

    g_preds = torch.tensor(g_preds).to(dtype=torch.float)
    print(f"accuracy: {g_preds.mean().item():.2f}")


forest()
