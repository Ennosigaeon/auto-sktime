import os
import shutil
from typing import Optional

import numpy as np
import pandas as pd


def evaluate_autogluon(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int,
        hpo: bool = True
):
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    if os.path.exists('./AutogluonModels'):
        shutil.rmtree('./AutogluonModels')

    if isinstance(y, pd.DataFrame):
        columns = y.columns.copy()
    else:
        columns = [y.name]
    y = y.reset_index()
    if 'series' not in y.columns:
        y['series'] = 0

    train_data = TimeSeriesDataFrame.from_data_frame(
        y,
        id_column="series",
        timestamp_column="index"
    )
    if X_train is not None:
        for col in X_train:
            train_data[col] = X_train[col].values
    if X_test is not None:
        X_test = X_test.reset_index()
        if 'series' not in X_test.columns:
            X_test['series'] = 0
        X_test = TimeSeriesDataFrame.from_data_frame(
            X_test,
            id_column="series",
            timestamp_column="index"
        )

    # Disable CUDA due to SIGSEV
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    res = []
    res_ints = []

    for col in columns:
        predictor = TimeSeriesPredictor(
            prediction_length=fh,
            quantile_levels=[0.25, 0.75],
            target=col,
            eval_metric="sMAPE",
            known_covariates_names=set(X_train.columns) - {'index', 'series'} if X_train is not None else None
        )

        predictor.fit(
            train_data,
            presets="best_quality" if hpo else None,
            time_limit=max_duration,
            random_seed=seed,
            hyperparameter_tune_kwargs={'num_trials': 10000, 'scheduler': 'local', 'searcher': 'auto'} if hpo else None
        )

        predictions = predictor.predict(train_data, known_covariates=X_test)
        predictions = predictions.reset_index(drop=True)

        y_pred_ints = pd.DataFrame(
            predictions[['0.25', '0.75']].values,
            columns=pd.MultiIndex.from_tuples([(f'Coverage_{col}', 0.5, 'lower'), (f'Coverage_{col}', 0.5, 'upper')]),
            index=predictions.index
        )

        res.append(predictions[['mean']].rename(columns={'mean': col}))
        res_ints.append(y_pred_ints)

    return pd.concat(res, axis=1), pd.concat(res_ints, axis=1)


def evaluate_autogluon_hpo(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int,
        hpo: bool = True
):
    return evaluate_autogluon(y, X_train, X_test, fh, max_duration, name, seed, hpo)
