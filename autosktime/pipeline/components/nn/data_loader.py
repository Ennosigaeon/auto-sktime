from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, \
    IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, COMPONENT_PROPERTIES
from autosktime.pipeline.components.nn.util import NN_DATA
from autosktime.pipeline.util import Int64Index
from autosktime.util.backend import ConfigContext, ConfigId


class TimeSeriesDataset(Dataset):

    def __init__(self, x: List[torch.Tensor], y: List[torch.Tensor]):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class SequenceDataLoaderComponent(AutoSktimeComponent):

    def __init__(
            self,
            batch_size: int = 1,
            validation_size: int = 0.2,
            sequence_length: int = None,
            random_state: np.random.RandomState = None,
            config_id: ConfigId = None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.validation_size = validation_size
        self.random_state = random_state
        self.config_id = config_id

    def fit(self, data: NN_DATA, y=None):
        X, y = data.get('X'), data.get('y')

        config: ConfigContext = ConfigContext.instance()
        self.sequence_length = max(config.get_config(self.config_id, 'panel_sizes', default=[0]))
        X_train, y_train = self._prepare_train_data(X, y)

        if 'X_val' in data:
            X_val, y_val = self._prepare_validation_data(data.get('X_val'), data.get('y_val'))
        elif self.validation_size > 0:
            y_train, y_val, X_train, X_val = train_test_split(y_train, X_train, test_size=self.validation_size,
                                                              random_state=self.random_state)
        else:
            X_val, y_val = X_train, y_train

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

    def _prepare_train_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._prepare_data(X, y, 'panel_sizes')

    def _prepare_validation_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        config: ConfigContext = ConfigContext.instance()
        key = 'panel_sizes_val' if 'panel_sizes_val' in config.get_config(self.config_id) else 'panel_sizes'
        return self._prepare_data(X, y, key)

    def _prepare_data(
            self,
            X: np.ndarray,
            y: np.ndarray,
            key: str = 'panel_sizes'
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if y is None:
            y = np.zeros(X.shape[0])

        config: ConfigContext = ConfigContext.instance()
        splits = np.cumsum(config.get_config(self.config_id, key, default=[0])[:-1])

        xs = np.split(X, splits)
        ys = np.split(y, splits)

        requires_padding = self.batch_size > 1 and not np.all([x.shape == xs[0].shape for x in xs])

        stacked_x = []
        stacked_y = []
        for seq_x, seq_y in zip(xs, ys):
            seq_x = torch.tensor(seq_x)
            seq_y = torch.tensor(seq_y)

            if requires_padding:
                padding_length = self.sequence_length - seq_x.shape[0]
                seq_x = torch.nn.functional.pad(seq_x, (0, 0, padding_length, 0))
                seq_y = torch.nn.functional.pad(seq_y, (padding_length, 0))
            stacked_x.append(seq_x.float())
            stacked_y.append(seq_y.float())

        return stacked_x, stacked_y

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
        batch_size = CategoricalHyperparameter('batch_size', [1], default_value=1)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([batch_size])

        return cs


class ChunkedDataLoaderComponent(SequenceDataLoaderComponent):

    def __init__(
            self,
            window_length: int = 30,
            batch_size: int = 128,
            validation_size: int = 0.2,
            sequence_length: int = None,
            random_state: np.random.RandomState = None,
            config_id: ConfigId = None
    ):
        super().__init__(batch_size, validation_size, sequence_length, random_state, config_id)
        self.window_length = window_length

    def _prepare_data(
            self,
            X: np.ndarray,
            y: np.ndarray,
            key: str = 'panel_sizes'
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if y is None:
            y = np.zeros(X.shape[0])

        config: ConfigContext = ConfigContext.instance()
        splits = np.cumsum(config.get_config(self.config_id, key, default=[0])[:-1])

        xs = np.split(X, splits)
        Xt = np.concatenate([self._generate_lookback(x_, self.window_length) for x_ in xs])

        return torch.tensor(Xt).float(), torch.tensor(y).float()

    @staticmethod
    def _generate_lookback(array: np.ndarray, window_length: int):
        start = -window_length + 1
        sub_windows = (
                start +
                np.expand_dims(np.arange(window_length), 0) +
                np.expand_dims(np.arange(array.shape[0]), 0).T
        )
        sub_windows[sub_windows < 0] = 0

        return array[sub_windows]

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        batch_size = CategoricalHyperparameter('batch_size', [128], default_value=128)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([batch_size])

        return cs
