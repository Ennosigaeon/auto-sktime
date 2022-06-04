import logging
from typing import List, Tuple, Any, Dict, Union

import numpy as np
import pandas as pd
import sklearn.utils
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


class KNearestDataSets:
    def __init__(
            self,
            logger: logging.Logger,
            metric: str = 'l1',
            random_state=None,
            metric_params: Dict[str, Any] = None
    ):
        self.logger = logger
        self.metric = metric
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.metric_params = metric_params if metric_params is not None else {}

        self.best_configuration_per_dataset_: Dict[str, Union[int, None]] = dict()

    # noinspection PyAttributeOutsideInit
    def fit(self, X: pd.DataFrame, runs: pd.DataFrame):
        if callable(self.metric):
            metric = self.metric
            p = 0
        elif self.metric.lower() == "l1":
            metric = "minkowski"
            p = 1
        elif self.metric.lower() == "l2":
            metric = "minkowski"
            p = 2
        else:
            raise ValueError(self.metric)

        self.n_data_sets_ = X.shape[0]
        self._prepare_payload(runs)

        self._scaler = MinMaxScaler()
        X_train = self._scaler.fit_transform(X)

        self._nearest_neighbors = NearestNeighbors(
            n_neighbors=self.n_data_sets_, radius=None, algorithm="brute",
            leaf_size=30, metric=metric, p=p,
            metric_params=self.metric_params)
        self._nearest_neighbors.fit(X_train)

        self.timeseries_ = X

    def _prepare_payload(self, runs: pd.DataFrame):
        # for each dataset, sort the runs according to their result
        for dataset_name in runs:
            if not np.isfinite(runs[dataset_name]).any():
                self.best_configuration_per_dataset_[dataset_name] = None
            else:
                configuration_idx = runs[dataset_name].index[np.nanargmin(runs[dataset_name].values)]
                self.best_configuration_per_dataset_[dataset_name] = configuration_idx

    def k_best_suggestions(self, x: pd.Series, k: int = 1, exclude_double_configurations: bool = True) -> List[
        Tuple[str, float, int]]:
        if k < -1 or k == 0:
            raise ValueError('Number of neighbors k cannot be zero or negative.')
        nearest_datasets, distances = self.kneighbors(x, self.n_data_sets_)

        kbest: List[Tuple[str, float, int]] = []

        added_configurations = set()
        for dataset_name, distance in zip(nearest_datasets, distances):
            best_configuration = self.best_configuration_per_dataset_[dataset_name]

            if best_configuration is None:
                self.logger.info(f'Found no best configuration for instance {dataset_name}')
                continue

            if exclude_double_configurations:
                if best_configuration not in added_configurations:
                    added_configurations.add(best_configuration)
                    kbest.append((dataset_name, distance, best_configuration))
            else:
                kbest.append((dataset_name, distance, best_configuration))

            if k != -1 and len(kbest) >= k:
                break

        if k == -1:
            k = len(kbest)
        return kbest[:k]

    def kneighbors(self, x: pd.Series, k: int = 1) -> Tuple[List[str], List[float]]:
        if k < -1 or k == 0:
            raise ValueError('Number of neighbors k cannot be zero or negative.')
        elif k == -1:
            k = self.n_data_sets_

        x = self._scaler.transform(x.to_frame().transpose())
        distances, neighbor_indices = self._nearest_neighbors.kneighbors(x, n_neighbors=k, return_distance=True)

        assert k == neighbor_indices.shape[1]

        # Neighbor indices is 2d, each row is the indices for one dataset in x.
        rval = [self.timeseries_.index[i] for i in neighbor_indices[0]]

        return rval, distances[0]
