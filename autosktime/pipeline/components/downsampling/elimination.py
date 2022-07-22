from typing import Union

import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES
from autosktime.pipeline.components.downsampling import BaseDownSampling
from autosktime.pipeline.components.downsampling.base import fix_size
from autosktime.pipeline.util import Int64Index


class EliminationDownSampler(BaseDownSampling):

    def __init__(self, window_size: Union[float, int] = 10, random_state: np.random.RandomState = None):
        super().__init__()
        self.window_size = window_size
        self.random_state = random_state

    def _transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.DataFrame = None):
        if isinstance(self.window_size, float):
            self.window_size_ = max(1, int(X.shape[0] * self.window_size))
        else:
            self.window_size_ = int(self.window_size)
        self._original_size = X.shape[0]

        Xt = pd.DataFrame(X.values[::self.window_size_], columns=X.columns, index=X.index[::self.window_size_])
        if y is not None:
            yt = pd.DataFrame(y.values[::self.window_size_], columns=y.columns, index=y.index[::self.window_size_])
        else:
            yt = None

        return Xt, yt

    def _inverse_transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        Xt = np.repeat(X.values, self.window_size_)
        Xt = fix_size(Xt, self._original_size)

        if y is not None:
            yt = np.repeat(y.values, self.window_size_)
            yt = fix_size(yt, self._original_size)
        else:
            yt = None
        return Xt, yt

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        window_size = UniformFloatHyperparameter('window_size', 0.001, 0.1, default_value=0.01)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([window_size])
        return cs

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }
