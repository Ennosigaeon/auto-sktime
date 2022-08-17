from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformIntegerHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, \
    IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, COMPONENT_PROPERTIES
from autosktime.pipeline.components.nn.util import NN_DATA
from autosktime.pipeline.util import Int64Index


class TimeSeriesDataset(Dataset):

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class DataLoaderComponent(AutoSktimeComponent):

    def __init__(
            self,
            batch_size: int = 8,
            window_length: int = 10,
            validation_size: int = 0.1,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.window_length = window_length
        self.validation_size = validation_size
        self.random_state = random_state

    def fit(self, data: NN_DATA, y=None):
        X, y = self._prepare_data(data.get('X'), data.get('y'))

        if self.validation_size > 0:
            y_train, y_val, X_train, X_val = train_test_split(y, X, test_size=self.validation_size,
                                                              random_state=self.random_state)
        else:
            y_train, y_val, X_train, X_val = y, y, X, X

        self.train_loader_ = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        self.val_loader_ = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)

        return self

    def transform(self, data: NN_DATA) -> NN_DATA:
        X, y = self._prepare_data(data.get('X'), data.get('y'))

        dataset = TimeSeriesDataset(X, y)
        data.update(
            {
                'train_data_loader': self.train_loader_,
                'val_data_loader': self.val_loader_,
                'test_data_loader': DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            }
        )
        return data

    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # By convention the last column contains the sequence id
        splits = np.where(X[:-1, -1] != X[1:, -1])[0] + 1

        xs = np.split(X[:, :-1], splits)
        X = np.concatenate([self._generate_lookback(x_, self.window_length) for x_ in xs])

        if y is None:
            y = np.zeros(X.shape[0])

        return X, y

    @staticmethod
    def _generate_lookback(array, window_length: int):
        start = -window_length + 1
        sub_windows = (
                start +
                np.expand_dims(np.arange(window_length), 0) +
                np.expand_dims(np.arange(array.shape[0]), 0).T
        )
        sub_windows[sub_windows < 0] = 0

        return array[sub_windows]

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        batch_size = CategoricalHyperparameter('batch_size', [16, 32, 64, 128, 256, 512], default_value=512)
        window_length = UniformIntegerHyperparameter('window_length', lower=10, upper=50, default_value=25)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([batch_size, window_length])

        return cs
