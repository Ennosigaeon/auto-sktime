import glob
import logging
import math
import pathlib
import time
import traceback
from typing import Callable

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('graphviz').setLevel(logging.WARNING)

import pandas as pd
from matplotlib import pyplot as plt
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.utils.plotting import plot_series

from scripts.benchmark._autogluon import evaluate_autogluon
from scripts.benchmark._autots import evaluate_autots
from scripts.benchmark._hyperts import evaluate_hyperts
from scripts.benchmark._pyaf import evaluate_pyaf
from scripts.benchmark._sktime import evaluate_arima, evaluate_prophet

methods = [
    ('pmdarima', evaluate_arima),
    ('prophet', evaluate_prophet),
    ('pyaf', evaluate_pyaf),
    ('hyperts', evaluate_hyperts),
    ('autogluon', evaluate_autogluon),
    ('autots', evaluate_autots)
]
fh = 12
max_duration = 60
metric = MeanAbsolutePercentageError(symmetric=True)

results = {'method': [], 'dataset': [], 'smape': [], 'duration': []}


def test_framework(y_train: pd.Series, y_test: pd.Series, evaluate: Callable):
    start = time.time()
    try:
        y_pred, y_pred_ints = evaluate(y_train.copy(), fh, max_duration)
        y_pred.index, y_pred_ints.index = y_test.index, y_test.index

        score = metric(y_test, y_pred)

        return y_pred, y_pred_ints, score, time.time() - start
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        return None, None, math.inf, time.time() - start


def benchmark():
    files = sorted(glob.glob('../data/univariate/real/*.csv'), key=str.casefold)
    for i, file in enumerate(files):
        path = pathlib.Path(file)
        if '_hourly' in path.name:
            continue

        print(f'{path.name} - {i}/{len(files)}')

        y = pd.read_csv(file, index_col='index', parse_dates=['index'])['y']
        y_train, y_test = temporal_train_test_split(y, test_size=fh)

        for name, evaluate in methods:
            y_pred, y_pred_ints, score, duration = test_framework(y_train, y_test, evaluate)
            if y_pred is not None:
                plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"], pred_interval=y_pred_ints,
                            title=f'{path.name} - {name}')
                plt.show()

            results['method'].append(name)
            results['dataset'].append(path.name)
            results['smape'].append(score)
            results['duration'].append(duration)

        break

    print(pd.DataFrame(results))


if __name__ == '__main__':
    benchmark()
