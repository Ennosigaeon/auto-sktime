import os
import shutil
from typing import Optional

import numpy as np
import pandas as pd

from scripts.benchmark.util import generate_fh


def evaluate_hyperts(
        y: pd.Series,
        X_train: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        fh: int,
        max_duration: int,
        name: str,
        seed: int
):
    from hyperts import make_experiment
    from hyperts.framework.compete import TSPipeline

    fh = pd.DataFrame(generate_fh(y.index, fh).to_timestamp(), columns=['index'])
    fh.index += y.shape[0]

    if X_test is not None:
        for col in X_test:
            fh[col] = X_test[col].values

    if X_train is not None:
        y = pd.concat((y, X_train), axis=1)

    y = y.reset_index()

    model: TSPipeline = make_experiment(
        y,
        task='univariate-forecast' if 'y' in y.columns else 'multivariate-forecast',
        timestamp='index',
        max_trials=500000,
        early_stopping_rounds=500000,
        early_stopping_time_limit=max_duration,
        random_state=seed,
        covariates=list(X_train.columns) if X_train is not None else None
    ).run()

    y_pred = model.predict(fh).set_index('index')
    y_pred_ints = pd.DataFrame(
        np.tile(y_pred.values, 2),
        columns=pd.MultiIndex.from_tuples(
            [(f'Coverage_{col}', 0.5, 'lower') for col in y_pred.columns] +
            [(f'Coverage_{col}', 0.5, 'upper') for col in y_pred.columns]
        ),
        index=y_pred.index
    )

    if os.path.exists('/tmp/workdir'):
        shutil.rmtree('/tmp/workdir')

    return y_pred, y_pred_ints
