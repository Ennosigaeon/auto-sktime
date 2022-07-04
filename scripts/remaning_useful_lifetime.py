import os.path
import shutil

import numpy as np
from autosktime.automl import AutoML
from autosktime.data.benchmark.rul import load_rul
from autosktime.data.splitter import multiindex_cross_validation
from autosktime.metrics import calculate_loss, RootMeanSquaredError
from autosktime.util import resolve_index
from autosktime.util.plotting import plot_series
from matplotlib import pyplot as plt
from sktime.forecasting.base import ForecastingHorizon

X, y = load_rul(os.path.join(__file__, 'data', 'rul'))
folds = 5

performance = np.zeros(folds)

for fold, (y_train, y_test, X_train, X_test) in \
        enumerate(multiindex_cross_validation(y, X, folds=folds, random_state=42)):
    workdir = f'fold_{fold}'
    try:
        shutil.rmtree(workdir)
    except FileNotFoundError:
        pass

    automl = AutoML(
        time_left_for_this_task=60,
        per_run_time_limit=30,
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

    X_test[y_test.columns[0]] = y_test
    y_pred = automl.predict(ForecastingHorizon(resolve_index(y_test.index), is_relative=False), X_test)

    performance[fold] = calculate_loss(y_test, y_pred, automl._task, automl._metric)

    plot_series(None, y_test, y_pred)
    plt.show()

print('Performance', performance, 'Mean', np.mean(performance))
