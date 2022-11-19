import json
import os.path
import pickle
import shutil
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sktime.forecasting.base import ForecastingHorizon

from autosktime.automl import AutoML
from autosktime.constants import PANEL_INDIRECT_FORECAST
from autosktime.data.benchmark import *
from autosktime.data.splitter import multiindex_split
from autosktime.metrics import RootMeanSquaredError
from autosktime.util import resolve_index
from autosktime.util.plotting import plot_grouped_series

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data_dir = os.path.join(Path(__file__).parent.resolve(), 'data', 'rul')
# benchmark = RULBenchmark(count=10)
benchmark = CMAPSSBenchmark(number=1)

X, y = benchmark.get_data()
train_folds, val_folds, test_folds = benchmark.get_train_test_splits()

for fold, ((_, train), (_, val), (_, test)) in enumerate(
        zip(train_folds.iterrows(), val_folds.iterrows(), test_folds.iterrows())):
    y_train, y_test, X_train, X_test = multiindex_split(y, X, train=pd.concat((train, val)), test=test)
    workdir = f'rul/fold_{fold}'
    try:
        shutil.rmtree(workdir)
    except FileNotFoundError:
        pass

    automl = AutoML(
        time_left_for_this_task=300,
        per_run_time_limit=60,
        ensemble_size=10,
        ensemble_nbest=50,
        n_jobs=1,
        seed=fold,
        temporary_directory=workdir,
        metric=RootMeanSquaredError(start=0.1),
        resampling_strategy='panel-pre',
        resampling_strategy_arguments={'train_ids': [pd.concat((train, val))], 'test_ids': [test]},
        delete_tmp_folder_after_terminate=False,
        use_pynisher=False,
        use_multi_fidelity=True,
        verbose=True
    )

    automl.fit(y, X, dataset_name='rul', task=PANEL_INDIRECT_FORECAST)

    y_pred = automl.predict(ForecastingHorizon(resolve_index(y_test.index), is_relative=False), X_test)
    benchmark.score_solutions(y_pred, y_test)

    plot_grouped_series(None, y_test, y_pred)
    plt.savefig(os.path.join(workdir, 'plot.pdf'))

    with open(os.path.join(workdir, 'model.pkl'), 'wb') as f:
        pickle.dump(automl, f)
    with open(os.path.join(workdir, 'ensemble.json'), 'w') as f:
        json.dump(automl.ensemble_configurations_, f)

    df = pd.DataFrame(benchmark.performance)
    print(df)
    print(df.describe())

    break
