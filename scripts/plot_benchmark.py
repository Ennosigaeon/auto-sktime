import math

import pandas as pd
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from autosktime.data.benchmark import BENCHMARKS
from autosktime.data.splitter import multiindex_split

parser = ArgumentParser()
parser.add_argument('benchmark', type=str)

args = parser.parse_args()

benchmark = BENCHMARKS[args.benchmark]()
X, y = benchmark.get_data()
train_folds, val_folds, test_folds = benchmark.get_train_test_splits()

for (_, train), (_, val), (_, test) in zip(train_folds.iterrows(), val_folds.iterrows(), test_folds.iterrows()):
    y_train, y_test, X_train, X_test = multiindex_split(y, X, train=pd.concat((train, val)), test=test)

    fig, axes = plt.subplots(int(math.ceil(len(X_train.columns) / 2)), 2, figsize=(20, 20))
    fig.suptitle(benchmark.name(), fontsize=16)
    axes = axes.flatten()
    keys = X_train.index.remove_unused_levels().levels[0]

    for i, col in enumerate(X_train):
        axes[i].set_title(col)
        for idx in keys:
            X_ = X_train.loc[idx, col]
            axes[i].plot(X_)

    plt.show()

    break
