import argparse
import logging
import math
import os
import time
import traceback
from datetime import datetime
from typing import Callable, Optional

from autosktime.data.benchmark.timeseries import load_timeseries
from autosktime.metrics import MeanAbsoluteScaledError
from scripts.benchmark._autopytorch import evaluate_autopytorch
from scripts.benchmark._deepar import evaluate_deepar
from scripts.benchmark._sktime import evaluate_naive, evaluate_ets
from scripts.benchmark._tft import evaluate_temporal_fusion_transformer

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('graphviz').setLevel(logging.WARNING)

import pandas as pd
from matplotlib import pyplot as plt
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
    'ets': evaluate_ets,
    'pmdarima': evaluate_arima,
    # 'prophet': evaluate_prophet,
    'naive': evaluate_naive,
    'tft': evaluate_temporal_fusion_transformer,
    'deepar': evaluate_deepar,
    'pyaf': evaluate_pyaf,
    'hyperts': evaluate_hyperts,
    'autogluon': evaluate_autogluon,
    # 'autogluon_hpo': evaluate_autogluon_hpo,
    # 'autots': evaluate_autots,
    'autots_random': evaluate_autots_random,
    'auto-pytorch': evaluate_autopytorch
}

parser = argparse.ArgumentParser()
parser.add_argument('--fh', type=int, default=None)
parser.add_argument('--max_duration', type=int, default=300)
parser.add_argument('--repetitions', type=int, default=5)
parser.add_argument('--method', type=str, choices=methods.keys(), default=None)
parser.add_argument('--start_index', type=int, default=None)
parser.add_argument('--end_index', type=int, default=None)
args = parser.parse_args()

metric = MeanAbsoluteScaledError()

results = {'method': [], 'seed': [], 'dataset': [], 'mase': [], 'duration': []}


def test_framework(
        y_train: pd.Series,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        evaluate: Callable,
        fh: int,
        **kwargs
):
    start = time.time()
    try:
        y_pred, y_pred_ints = evaluate(
            y_train.copy(),
            X_train.copy() if X_train is not None else None,
            X_test.copy() if X_test is not None else None,
            fh,
            args.max_duration,
            **kwargs
        )
        y_pred.index, y_pred_ints.index = y_test.index, y_test.index

        score = metric(y_test, y_pred, y_train=y_train)

        return y_pred, y_pred_ints, score, time.time() - start
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        return None, None, math.inf, time.time() - start


def benchmark(plot: bool = False):
    result_file = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(f'results/{result_file}')

    for i, (y_train, y_test, X_train, X_test, ds_name, fh) in enumerate(load_timeseries(args.fh)):
        if (args.start_index is not None and i < args.start_index) or \
                (args.end_index is not None and i >= args.end_index):
            continue
        print(ds_name, fh)

        for name, evaluate in methods.items():
            if args.method is not None and name != args.method:
                continue

            for seed in range(args.repetitions):
                y_pred, y_pred_ints, score, duration = test_framework(
                    y_train, y_test, X_train, X_test, evaluate, fh,
                    name=ds_name, seed=seed
                )
                if y_pred is not None and plot:
                    fig, ax = plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"],
                                          pred_interval=y_pred_ints, title=f'{ds_name} - {name}')
                    plt.show()
                    plt.close(fig)

                results['method'].append(name)
                results['seed'].append(seed)
                results['dataset'].append(ds_name)
                results['mase'].append(score)
                results['duration'].append(duration)

                pd.DataFrame(results).to_csv(f'results/{result_file}/_result.csv', index=False)
                if y_pred is not None:
                    y_pred.to_csv(f'results/{result_file}/{ds_name[:-4]}-{name}-{seed}-prediction.csv')

    print(pd.DataFrame(results))


if __name__ == '__main__':
    benchmark()
