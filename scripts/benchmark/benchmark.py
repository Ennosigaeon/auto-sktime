import argparse
import glob
import logging
import math
import pathlib
import time
import traceback
from datetime import datetime
from typing import Callable, Optional

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('graphviz').setLevel(logging.WARNING)

import pandas as pd
from matplotlib import pyplot as plt
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.utils.plotting import plot_series

from _autogluon import evaluate_autogluon, evaluate_autogluon_hpo
from _autosktime import evaluate_autosktime, evaluate_autosktime_multi_fidelity, \
    evaluate_autosktime_warm_starting, evaluate_autosktime_templates
from _autots import evaluate_autots, evaluate_autots_random
from _hyperts import evaluate_hyperts
from _pyaf import evaluate_pyaf
from _sktime import evaluate_arima, evaluate_prophet

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 10000)
pd.set_option("display.precision", 4)

methods = {
    'auto-sktime_templates': evaluate_autosktime_templates,
    'auto-sktime_multi_fidelity': evaluate_autosktime_multi_fidelity,
    'auto-sktime_warm_starting': evaluate_autosktime_warm_starting,
    'auto-sktime': evaluate_autosktime,
    'pmdarima': evaluate_arima,
    'prophet': evaluate_prophet,
    'pyaf': evaluate_pyaf,
    'hyperts': evaluate_hyperts,
    'autogluon': evaluate_autogluon,
    'autogluon_hpo': evaluate_autogluon_hpo,
    'autots': evaluate_autots,
    'autots_random': evaluate_autots_random,
}

parser = argparse.ArgumentParser()
parser.add_argument('--fh', type=int, default=12)
parser.add_argument('--max_duration', type=int, default=60)
parser.add_argument('--repetitions', type=int, default=5)
parser.add_argument('--method', type=str, choices=methods.keys(), default=None)
parser.add_argument('--start_index', type=int, default=None)
parser.add_argument('--end_index', type=int, default=None)
args = parser.parse_args()

metric = MeanAbsolutePercentageError(symmetric=True)

results = {'method': [], 'seed': [], 'dataset': [], 'smape': [], 'duration': []}


def test_framework(
        y_train: pd.Series,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        evaluate: Callable,
        **kwargs
):
    start = time.time()
    try:
        y_pred, y_pred_ints = evaluate(
            y_train.copy(),
            X_train.copy() if X_train is not None else None,
            X_test.copy() if X_test is not None else None,
            args.fh,
            args.max_duration,
            **kwargs
        )
        y_pred.index, y_pred_ints.index = y_test.index, y_test.index

        score = metric(y_test, y_pred)

        return y_pred, y_pred_ints, score, time.time() - start
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        return None, None, math.inf, time.time() - start


def benchmark(plot: bool = False):
    result_file = datetime.now().strftime("%Y%m%d-%H%M%S")
    files = sorted(glob.glob('../data/univariate/real/*.csv'), key=str.casefold)
    for i, file in enumerate(files):
        if (args.start_index is not None and i < args.start_index) or \
                (args.end_index is not None and i >= args.end_index):
            continue

        path = pathlib.Path(file)

        print(f'{path.name} - {i}/{len(files)}')

        df = pd.read_csv(file, parse_dates=['index'])
        if 'series' in df.columns:
            pass
        else:
            df.index = df['index']
            df = df.drop(columns=['index'])

            if 'y' in df.columns:
                y = df['y']
                y_train, y_test = temporal_train_test_split(y, test_size=args.fh)
                if len(df.columns) == 1:
                    X_train, X_test = None, None
                else:
                    df = df.drop(columns=['y'])
                    X_train, X_test = temporal_train_test_split(df, test_size=args.fh)
            else:
                y_train, y_test = temporal_train_test_split(df, test_size=args.fh)
                X_train, X_test = None, None

        for name, evaluate in methods.items():
            if args.method is not None and name != args.method:
                continue

            for seed in range(args.repetitions):
                y_pred, y_pred_ints, score, duration = test_framework(
                    y_train, y_test, X_train, X_test, evaluate,
                    name=path.name, seed=seed
                )
                if y_pred is not None and plot:
                    fig, ax = plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"],
                                          pred_interval=y_pred_ints, title=f'{path.name} - {name}')
                    plt.show()
                    plt.close(fig)

                results['method'].append(name)
                results['seed'].append(seed)
                results['dataset'].append(path.name)
                results['smape'].append(score)
                results['duration'].append(duration)

                pd.DataFrame(results).to_csv(f'results/results-{result_file}.csv', index=False)

    print(pd.DataFrame(results))


if __name__ == '__main__':
    benchmark()
