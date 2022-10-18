from typing import Union

import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeTransformer
from autosktime.pipeline.util import Int64Index, NotVectorizedMixin


class TargetMinMaxScalerComponent(NotVectorizedMixin, AutoSktimeTransformer):

    def __init__(
            self,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.random_state = random_state

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        from sklearn.preprocessing import MinMaxScaler
        self.estimator = MinMaxScaler(copy=False)

        self.estimator.fit(X=X, y=y)
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        if self.estimator is None:
            raise NotImplementedError
        Xt = self.estimator.transform(X)
        if isinstance(X, pd.Series):
            return pd.Series(Xt, index=X.index, name=X.name)
        else:
            return pd.DataFrame(Xt, index=X.index, columns=X.columns)

    # noinspection PyUnresolvedReferences
    def inverse_transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        if self.estimator is None:
            raise NotImplementedError()
        Xt = self.estimator.inverse_transform(X)
        if isinstance(X, pd.Series):
            return pd.Series(Xt, index=X.index, name=X.name)
        else:
            return pd.DataFrame(Xt, index=X.index, columns=X.columns)

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: True,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs
