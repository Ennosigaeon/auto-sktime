import os.path
import warnings

import json
import pandas as pd
import pickle
import shutil
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from sktime.forecasting.base import ForecastingHorizon

from autosktime.automl import AutoML
from autosktime.constants import PANEL_INDIRECT_FORECAST
from autosktime.data.benchmark import BENCHMARKS
from autosktime.data.splitter import multiindex_split
from autosktime.metrics import RootMeanSquaredError
from autosktime.util import resolve_index
from autosktime.util.arg_types import fold_type, parse_folds
from autosktime.util.plotting import plot_grouped_series

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

parser = ArgumentParser()
parser.add_argument('benchmark', type=str, help='The benchmark to run', choices=BENCHMARKS.keys())
parser.add_argument('--runtime', type=int, default=36000, help='Total optimization time in seconds')
parser.add_argument('--timeout', type=int, default=300, help='Timeout to fit a single configuration in seconds')
parser.add_argument('--folds', type=fold_type, default='*',
                    help='Number of fold to run. Must be \'*\', an integer or a range like \'0-5\'')
parser.add_argument('--cleanup', type=bool, default=False, help='Delete results of prior executions')

args = parser.parse_args()

benchmark = BENCHMARKS[args.benchmark]()
folds = parse_folds(args.folds, benchmark.folds)

X, y = benchmark.get_data()
train_folds, val_folds, test_folds = benchmark.get_train_test_splits()

for fold, ((_, train), (_, val), (_, test)) in enumerate(
        zip(train_folds.iterrows(), val_folds.iterrows(), test_folds.iterrows())):
    if fold not in folds:
        continue

    y_train, y_test, X_train, X_test = multiindex_split(y, X, train=pd.concat((train, val)), test=test)
    workdir = f'results/{benchmark.name()}/fold_{fold}'
    try:
        if args.cleanup:
            shutil.rmtree(workdir)
            shutil.rmtree(workdir + '__tmp__')
    except FileNotFoundError:
        pass

    automl = AutoML(
        time_left_for_this_task=args.runtime,
        per_run_time_limit=args.timeout,
        ensemble_size=10,
        ensemble_nbest=50,
        n_jobs=1,
        seed=fold,
        working_directory=workdir,
        metric=RootMeanSquaredError(start=benchmark.start),
        resampling_strategy='panel-pre',
        resampling_strategy_arguments={'train_ids': [train], 'test_ids': [val]},
        delete_tmp_folder_after_terminate=True,
        use_pynisher=True,
        use_multi_fidelity=True,
        verbose=True
    )

    automl.fit(y, X, dataset_name='rul', task=PANEL_INDIRECT_FORECAST, y_test=y_test, X_test=X_test)

    with open(os.path.join(workdir, 'model.pkl'), 'wb') as f:
        pickle.dump(automl, f)
    with open(os.path.join(workdir, 'ensemble.json'), 'w') as f:
        json.dump(automl.ensemble_configurations_, f)

    y_pred = automl.predict(ForecastingHorizon(resolve_index(y_test.index), is_relative=False), X_test)
    benchmark.score_solutions(y_pred, y_test)
    with open(os.path.join(workdir, 'predictions.npy'), 'wb') as f:
        y_pred.to_pickle(f)

    plot_grouped_series(None, y_test, y_pred)
    plt.savefig(os.path.join(workdir, 'plot.pdf'))

    df = pd.DataFrame(benchmark.performance)
    automl._logger.info(f'Fold results\n{df}\n{df.describe()}')