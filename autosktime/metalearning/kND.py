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

        self.best_configuration_per_dataset_: Dict[str, Union[int, None]] = dict()

    # noinspection PyAttributeOutsideInit
    def fit(self, X: pd.DataFrame, runs: pd.DataFrame):
        dt = np.zeros((1, 1))
        self.metric_callable_ = _resolve_metric_to_factory(self.metric, dt, dt, _METRIC_INFOS)

        self._prepare_payload(runs)

        self.n_data_sets_ = X.shape[1]
        self.timeseries_ = [(np.atleast_2d(X[name].dropna()), name) for name in X]

    def _prepare_payload(self, runs: pd.DataFrame):
        # for each dataset, sort the runs according to their result
        for dataset_name in runs:
            if not np.isfinite(runs[dataset_name]).any():
                self.best_configuration_per_dataset_[dataset_name] = None
            else:
                configuration_idx = runs[dataset_name].index[np.nanargmin(runs[dataset_name].values)]
                self.best_configuration_per_dataset_[dataset_name] = configuration_idx

    def k_best_suggestions(self, x: pd.Series, k: int = 1) -> List[Tuple[str, float, int]]:
        if k < -1 or k == 0:
            raise ValueError('Number of neighbors k cannot be zero or negative.')
        nearest_datasets, distances, idx = self.kneighbors(x, self.n_data_sets_)

        kbest: List[Tuple[str, float, int]] = []

        added_configurations = set()
        for dataset_name, distance in zip(nearest_datasets, distances):
            best_configuration = self.best_configuration_per_dataset_[dataset_name]

            if best_configuration is None:
                self.logger.info(f'Found no best configuration for instance {dataset_name}')
                continue

            kbest.append((dataset_name, distance, best_configuration))

            if k != -1 and len(kbest) >= k:
                break

        if k == -1:
            k = len(kbest)
        return kbest[:k]

    def kneighbors(self, x: pd.Series, k: int = 1) -> Tuple[List[str], List[float], List[int]]:
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
        return names[idx], distances[idx], idx
