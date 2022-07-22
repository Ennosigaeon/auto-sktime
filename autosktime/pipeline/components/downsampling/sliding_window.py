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


class SlidingWindowDownSampler(BaseDownSampling):

    def __init__(
            self,
            window_size: Union[float, int] = 0.01,
            overlap: float = 0.5,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.window_size = window_size
        self.overlap = overlap
        self.random_state = random_state

    def _transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.DataFrame = None):
        if isinstance(self.window_size, float):
            n = int(X.shape[0] * self.window_size)
        else:
            n = int(self.window_size)

        self._original_size = X.shape[0]

        Xt = self._window(X, n, self.overlap)

        index = X.index
        if isinstance(index, pd.PeriodIndex):
            index = pd.date_range(start=index[0].to_timestamp(), end=index[-1].to_timestamp(), periods=Xt.shape[0])
        else:
            index = np.linspace(index[0], index[-1], Xt.shape[0], endpoint=False, dtype=int)

        Xt = pd.DataFrame(Xt, columns=X.columns, index=index)
        if y is not None:
            yt = self._window(y, 50, self.overlap)
            yt = pd.DataFrame(yt, columns=y.columns, index=index)
        else:
            yt = None

        return Xt, yt

    @staticmethod
    def _window(df: Union[pd.Series, pd.DataFrame], window_size: int = 4, overlap: float = 0.5, copy=True):
        if isinstance(df, pd.DataFrame):
            cols = []
            for col in df:
                series = SlidingWindowDownSampler._window(df[col], window_size, overlap, copy)
                cols.append(series.mean(axis=1))
            return np.array(cols).T

        a = df.values

        sh = (df.size - window_size + 1, window_size)
        st = a.strides * 2
        view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::int(overlap * window_size)]
        if copy:
            return view.copy()
        else:
            return view

    def _inverse_transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        Xt = np.repeat(X.values, self._X_filter.shape[0])
        Xt = fix_size(Xt, self._original_size)

        if y is not None:
            yt = np.repeat(y.values, self._y_filter.shape[0])
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
