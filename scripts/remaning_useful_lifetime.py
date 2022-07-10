import os.path
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sktime.forecasting.base import ForecastingHorizon

from autosktime.automl import AutoML
from autosktime.data.benchmark.rul import load_rul
from autosktime.data.splitter import multiindex_cross_validation
from autosktime.metrics import calculate_loss, RootMeanSquaredError, STRING_TO_METRIC
from autosktime.util import resolve_index
from autosktime.util.plotting import plot_grouped_series

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

X, y = load_rul(os.path.join(Path(__file__).parent.resolve(), 'data', 'rul'))
folds = 5

performance = {
    'rmse': np.zeros(folds),
    'wrmse': np.zeros(folds),
    'mae': np.zeros(folds),
    'wmae': np.zeros(folds),
    'me': np.zeros(folds),
    'std': np.zeros(folds),
    'maare': np.zeros(folds),
    'relph': np.zeros(folds),
    'phrate': np.zeros(folds),
    'cra': np.zeros(folds),
}

for fold, (y_train, y_test, X_train, X_test) in \
        enumerate(multiindex_cross_validation(y, X, folds=folds, random_state=42)):
    workdir = f'rul/fold_{fold}'
    try:
        shutil.rmtree(workdir)
    except FileNotFoundError:
        pass

    fig, ax = plt.subplots(2, 1)
    for key in X_train.index.levels[0]:
        X_train.loc[[key], ['Differenzdruck']].plot(ax=ax[0])
        y_train.loc[[key], :].plot(ax=ax[1])
    plt.show()

    automl = AutoML(
        time_left_for_this_task=300,
        per_run_time_limit=60,
        ensemble_size=5,
        ensemble_nbest=20,
        temporary_directory=workdir,
        metric=RootMeanSquaredError(),
        resampling_strategy='panel-holdout',
        resampling_strategy_arguments={'fh': 5},
        delete_tmp_folder_after_terminate=False,
        use_pynisher=False
    )

    automl.fit(y_train, X_train, dataset_name='rul')

    y_pred = automl.predict(ForecastingHorizon(resolve_index(y_test.index), is_relative=False), X_test)

    for metric_name in performance.keys():
        metric = STRING_TO_METRIC[metric_name]
        performance[metric_name][fold] = calculate_loss(y_test, y_pred, automl._task, metric)

    plot_grouped_series(None, y_test, y_pred)
    plt.savefig(os.path.join(workdir, 'plot.pdf'))

    with open(os.path.join(workdir, 'model.pkl'), 'wb') as f:
        pickle.dump(automl, f)

    df = pd.DataFrame(performance)
    print(df)
    print(df.describe())
