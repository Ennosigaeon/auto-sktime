from typing import Union

import numpy as np
import pandas as pd
from scipy import signal

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES
from autosktime.pipeline.components.downsampling.base import BaseDownSampling
from autosktime.pipeline.util import Int64Index


class ResamplingDownSampling(BaseDownSampling):

    def __init__(
            self,
            num: Union[float, int] = 0.5,
            initial_size: int = None,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.num = num
        self.initial_size = initial_size
        self.random_state = random_state

    def _transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.DataFrame = None):
        self.initial_size = X.shape[0]

        if isinstance(self.num, float):
            n = int(self.initial_size * self.num)
        else:
            n = int(self.initial_size / self.num)

        index = X.index
        if isinstance(index, pd.PeriodIndex):
            index = pd.date_range(start=index[0].to_timestamp(), end=index[-1].to_timestamp(), periods=n)
        else:
            index = np.linspace(index[0], index[-1], n, endpoint=False, dtype=int)

        Xt = pd.DataFrame(signal.resample(X, n), columns=X.columns, index=index)
        if y is not None:
            yt = pd.DataFrame(signal.resample(y, n), columns=y.columns, index=index)
        else:
            yt = None
        return Xt, yt

    def _inverse_transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        Xt = signal.resample(X, self.initial_size)
        if y is not None:
            yt = signal.resample(y, self.initial_size)
        else:
            yt = None
        return Xt, yt

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        num = UniformFloatHyperparameter('num', 0.01, 1, default_value=0.5)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([num])
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
