from typing import Union

import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeTransformer


class ImputerComponent(AutoSktimeTransformer):
    from sktime.transformations.series.impute import Imputer

    _estimator_class = Imputer

    def __init__(self, method: str = 'drift', random_state: np.random.RandomState = None):
        super().__init__()
        self.method = method
        self.random_state = random_state

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        self.estimator = self._estimator_class(method=self.method, random_state=self.random_state)
        self.estimator.fit(X=X, y=y)
        return self

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: True,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, pd.core.indexes.numeric.Int64Index]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        method = CategoricalHyperparameter('method', [
            'drift', 'linear', 'nearest', 'mean', 'median', 'bfill', 'ffill', 'random'
        ], default_value='drift')

        cs = ConfigurationSpace()
        cs.add_hyperparameters([method])
        return cs
