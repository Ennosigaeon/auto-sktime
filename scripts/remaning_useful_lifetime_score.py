import json
import os.path
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from autosktime.automl import AutoML
from autosktime.constants import PANEL_INDIRECT_FORECAST
from autosktime.data.benchmark.rul import load_rul, load_rul_splits
from autosktime.data.splitter import multiindex_split
from autosktime.metrics import RootMeanSquaredError, STRING_TO_METRIC
from autosktime.util import resolve_index

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

with open('ensemble.json', 'r') as f:
    configs: List[Tuple[float, float, Dict]] = json.load(f)

data_dir = os.path.join(Path(__file__).parent.resolve(), 'data', 'rul')
X, y = load_rul(data_dir)
train_folds, val_folds, test_folds = load_rul_splits(data_dir, 200)

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
    workdir = f'rul/score'
    try:
        shutil.rmtree(workdir)
    except FileNotFoundError:
        pass

    automl = AutoML(
        time_left_for_this_task=10,  # Not used actually
        per_run_time_limit=10,  # Not used actually
        temporary_directory=workdir,
        metric=RootMeanSquaredError(),
        resampling_strategy='panel-pre',
        resampling_strategy_arguments={'train_ids': [train], 'test_ids': [val]},
        delete_tmp_folder_after_terminate=False,
        use_pynisher=False,
        use_multi_fidelity=False
    )

    automl.fit(y_train, X_train, dataset_name='rul', task=PANEL_INDIRECT_FORECAST, configs=configs)

    y_pred = automl.predict(ForecastingHorizon(resolve_index(y_test.index), is_relative=False), X_test)

    for metric_name in performance.keys():
        metric = STRING_TO_METRIC[metric_name](start=50)
        performance[metric_name][fold] = metric(y_test, y_pred)


    def export_results(idx: int):
        series = y_pred.loc[idx]
        series['Time'] = series.index / 10
        series['Error'] = y_pred.loc[idx, 'RUL'] - y_test.loc[idx, 'RUL']
        series.to_csv(os.path.join('rul', f'{idx}.csv'), index=False)


    if fold == 18:
        export_results(25)
        export_results(44)

df = pd.DataFrame(performance)
print(df)
print(df.describe())
