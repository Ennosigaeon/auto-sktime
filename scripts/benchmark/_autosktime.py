import shutil
from typing import Optional

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from scripts.benchmark.util import generate_fh, fix_frequency


def evaluate_autosktime(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int,
        multi_fidelity: bool = True,
        warm_starting: bool = True
):
    from autosktime.automl import AutoML
    from autosktime.constants import Budget
    from autosktime.metrics import MeanAbsolutePercentageError

    fix_frequency(y, X_train, X_test)
    fh_ = ForecastingHorizon(generate_fh(y.index, fh), is_relative=False)

    workdir = f'results/{name}_{multi_fidelity}_{warm_starting}/fold_{seed}'
    try:
        shutil.rmtree(workdir)
        shutil.rmtree(workdir + '__tmp__')
    except FileNotFoundError:
        pass

    automl = AutoML(
        time_left_for_this_task=max_duration,
        per_run_time_limit=60,
        runcount_limit=10000,
        ensemble_size=10,
        ensemble_nbest=50,
        n_jobs=1,
        seed=seed,
        working_directory=workdir,
        metric=MeanAbsolutePercentageError(),
        delete_tmp_folder_after_terminate=True,
        use_pynisher=True,
        budget=Budget.SeriesLength if multi_fidelity else None,
        hp_priors=warm_starting,
        refit=True,
        verbose=True,
        cache_dir=f'results/.cache/auto-sktime/'
    )

    automl.fit(y, X_train, dataset_name=name)
    y_pred = automl.predict(fh_, X_test)
    # y_pred_ints = automl.predict_interval()
    y_pred_ints = pd.DataFrame(
        np.tile(y_pred.values, (2, 1)).T,
        columns=pd.MultiIndex.from_tuples([('Coverage', 0.5, 'lower'), ('Coverage', 0.5, 'upper')]),
        index=y_pred.index
    )

    return y_pred, y_pred_ints


def evaluate_autosktime_templates(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int
):
    return evaluate_autosktime(y, X_train, X_test, fh, max_duration, name, seed, multi_fidelity=False,
                               warm_starting=False)


def evaluate_autosktime_warm_starting(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int
):
    return evaluate_autosktime(y, X_train, X_test, fh, max_duration, name, seed, multi_fidelity=False,
                               warm_starting=True)


def evaluate_autosktime_multi_fidelity(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int
):
    return evaluate_autosktime(y, X_train, X_test, fh, max_duration, name, seed, multi_fidelity=True,
                               warm_starting=False)
