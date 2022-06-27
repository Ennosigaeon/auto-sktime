from typing import Union

import pandas as pd

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeTransformer


class DifferencerComponent(AutoSktimeTransformer):
    from sktime.transformations.series.difference import Differencer

    _estimator_class = Differencer

    def __init__(self, lags: int = 1, random_state=None):
        super().__init__()
        self.lags = lags
        self.random_state = random_state

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        self.estimator = self._estimator_class(lags=self.lags)
        self.estimator.fit(X=X, y=y)
        return self

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: True,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        lags = CategoricalHyperparameter('lags', [1, 2, 4, 7, 12])

        cs = ConfigurationSpace()
        cs.add_hyperparameters([lags])
        return cs
