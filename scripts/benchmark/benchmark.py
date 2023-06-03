import glob
import pathlib
import time

import pandas as pd
from matplotlib import pyplot as plt
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.utils.plotting import plot_series

from scripts.benchmark._autogluon import evaluate_autogluon
from scripts.benchmark._hyperts import evaluate_hyperts
from scripts.benchmark._pyaf import evaluate_pyaf
from scripts.benchmark._sktime import evaluate_arima, evaluate_prophet

methods = [
    ('pmdarima', evaluate_arima),
    ('prophet', evaluate_prophet),
    ('pyaf', evaluate_pyaf),
    ('hyperts', evaluate_hyperts),
    ('autogluon', evaluate_autogluon)
]
fh = 12

results = {'method': [], 'dataset': [], 'smape': [], 'duration': []}
i = 0
for file in sorted(glob.glob('../data/univariate/real/*.csv'), key=str.casefold):
    path = pathlib.Path(file)
    if '_hourly' in path.name:
        continue

    print(path.name)

    y = pd.read_csv(file, index_col='Index')
    # TODO reset index should not be necessary
    y = y.reset_index()['y']

    y_train, y_test = temporal_train_test_split(y, test_size=fh)
    metric = MeanAbsolutePercentageError(symmetric=True)

    for name, evaluate in methods:
        start = time.time()
        y_pred, y_pred_ints = evaluate(y_train.copy(), fh)
        duration = time.time() - start
        score = metric(y_test, y_pred)

        plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"], pred_interval=y_pred_ints,
                    title=f'{path.name} - {name}')
        plt.show()

        results['method'].append(name)
        results['dataset'].append(path.name)
        results['smape'].append(score)
        results['duration'].append(duration)

    i += 1
    if i > 2:
        break

print(pd.DataFrame(results))
