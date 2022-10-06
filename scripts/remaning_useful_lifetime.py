import json
import os.path
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sktime.forecasting.base import ForecastingHorizon

from autosktime.automl import AutoML
from autosktime.constants import PANEL_INDIRECT_FORECAST
from autosktime.data.benchmark.rul import load_rul, load_rul_splits
from autosktime.data.splitter import multiindex_split
from autosktime.metrics import RootMeanSquaredError, STRING_TO_METRIC
from autosktime.util import resolve_index
from autosktime.util.plotting import plot_grouped_series

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data_dir = os.path.join(Path(__file__).parent.resolve(), 'data', 'rul')
X, y = load_rul(data_dir)
train_folds, val_folds, test_folds = load_rul_splits(data_dir, 10)

performance = {
    'rmse': np.zeros(train_folds.shape[0]),
    'wrmse': np.zeros(train_folds.shape[0]),
    'mae': np.zeros(train_folds.shape[0]),
    'wmae': np.zeros(train_folds.shape[0]),
    'me': np.zeros(train_folds.shape[0]),
    'std': np.zeros(train_folds.shape[0]),
    'maare': np.zeros(train_folds.shape[0]),
    'relph': np.zeros(train_folds.shape[0]),
    'phrate': np.zeros(train_folds.shape[0]),
    'cra': np.zeros(train_folds.shape[0]),
}

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
        metric=RootMeanSquaredError(start=125),
        resampling_strategy='panel-pre',
        resampling_strategy_arguments={'train_ids': [pd.concat((train, val))], 'test_ids': [test]},
        delete_tmp_folder_after_terminate=False,
        use_pynisher=False,
        use_multi_fidelity=True,
        verbose=True
    )

    automl.fit(y, X, dataset_name='rul', task=PANEL_INDIRECT_FORECAST)

    y_pred = automl.predict(ForecastingHorizon(resolve_index(y_test.index), is_relative=False), X_test)

    for metric_name in performance.keys():
        metric = STRING_TO_METRIC[metric_name](start=125)
        performance[metric_name][fold] = metric(y_test, y_pred)

    plot_grouped_series(None, y_test, y_pred)
    plt.savefig(os.path.join(workdir, 'plot.pdf'))

    with open(os.path.join(workdir, 'model.pkl'), 'wb') as f:
        pickle.dump(automl, f)
    with open(os.path.join(workdir, 'ensemble.json'), 'w') as f:
        json.dump(automl.ensemble_configurations_, f)

    df = pd.DataFrame(performance)
    print(df)
    print(df.describe())
