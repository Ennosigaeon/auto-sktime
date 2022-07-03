from typing import Union

import numpy as np
import pandas as pd
from scipy import signal

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES
from autosktime.pipeline.components.downsampling import BaseDownSampling
from autosktime.pipeline.components.downsampling.base import fix_size
from autosktime.pipeline.util import Int64Index


class ConvolutionDownSampler(BaseDownSampling):

    def __init__(self, window_size: Union[float, int] = 0.05):
        super().__init__()
        self.window_size = window_size

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        pass

    def _transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        # TODO X.shape[0] has to be multiple of self.window_size

        if isinstance(self.window_size, float):
            n = int(X.shape[0] * self.window_size)
        else:
            n = int(self.window_size)

        self._original_size = X.shape[0]
        self._filter = (1.0 / n) * np.ones(n)

        Xt = signal.convolve(X, self._filter, mode='valid')[::n]
        if y is not None:
            yt = signal.convolve(y, self._filter, mode='valid')[::n]
        else:
            yt = None

        return Xt, yt

    def _inverse_transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        Xt = np.repeat(X.values, self._filter.shape[0])
        Xt = fix_size(Xt, self._original_size)

        if y is not None:
            yt = np.repeat(y.values, self._filter.shape[0])
            yt = fix_size(yt, self._original_size)
        else:
            yt = None
        return Xt, yt

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        window_size = UniformFloatHyperparameter('window_size', 0.01, 0.1, default_value=0.05)

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
