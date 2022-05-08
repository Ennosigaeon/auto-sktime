from typing import Tuple

import pandas as pd

from autosktime.data import AbstractDataManager
from autosktime.data.splitter import HoldoutSplitter

from sktime.forecasting.compose import EnsembleForecaster


# noinspection PyAbstractClass
class PrefittedEnsembleForecaster(EnsembleForecaster):

    def _fit(self, y, X=None, fh=None):
        self.forecasters_ = self.forecasters


def get_ensemble_targets(datamanager: AbstractDataManager, ensemble_size: float) -> Tuple[pd.Series, pd.DataFrame]:
    y = datamanager.y
    _, test = next(HoldoutSplitter(ensemble_size).split(y))

    return _get_by_index(y, datamanager.X, test)


def get_ensemble_train(datamanager: AbstractDataManager, ensemble_size: float) -> Tuple[pd.Series, pd.DataFrame]:
    y = datamanager.y

    train, _ = next(HoldoutSplitter(ensemble_size).split(y))
    return _get_by_index(y, datamanager.X, train)


def _get_by_index(y, X, idx):
    y_idx = y.iloc[idx]
    if X is None:
        X_idx = None
    else:
        X_idx = X.iloc[idx, :]

    return y_idx, X_idx
