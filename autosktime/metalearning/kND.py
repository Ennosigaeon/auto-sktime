import logging
from typing import List, Tuple, Dict, Union

import numpy as np
import pandas as pd
from sktime.distances._distance import _METRIC_INFOS
from sktime.distances._resolve_metric import _resolve_metric_to_factory


class KNearestDataSets:
    def __init__(
            self,
            logger: logging.Logger,
            metric: str = 'ddtw',
    ):
        self.logger = logger
        self.metric = metric

    # noinspection PyAttributeOutsideInit
    def fit(self, X: pd.DataFrame):
        dt = np.zeros((1, 1))
        self.metric_callable_ = _resolve_metric_to_factory(self.metric, dt, dt, _METRIC_INFOS)

        self.n_data_sets_ = X.shape[1]
        self.timeseries_ = [(np.atleast_2d(X[name].dropna()), name) for name in X]

    def kneighbors(self, x: pd.Series, k: int = 1) -> Tuple[List[str], List[float]]:
        if k < -1 or k == 0:
            raise ValueError('Number of neighbors k cannot be zero or negative.')
        elif k == -1:
            k = self.n_data_sets_

        x = np.atleast_2d(x)

        distances = np.empty(self.n_data_sets_)
        names = np.empty(self.n_data_sets_, dtype='object')
        for i, (y, name) in enumerate(self.timeseries_):
            distances[i] = self.metric_callable_(x, y)
            names[i] = name

        idx = np.argsort(distances)[:k]
        return names[idx], distances[idx]
