from typing import Union

import pandas as pd

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeTransformer, AutoSktimePredictor


class ImputerComponent(AutoSktimeTransformer, AutoSktimePredictor):
    from sktime.transformations.series.impute import Imputer

    _estimator_class = Imputer

    def __init__(self, method: str = 'drift', random_state=None):
        super().__init__()
        self.method = method
        self.random_state = random_state

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        self.estimator = self._estimator_class(method=self.method)
        self.estimator.fit(X=X, y=y)
        return self

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: False,
            IGNORES_EXOGENOUS_X: True,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        method = CategoricalHyperparameter('method', [
            'drift', 'linear', 'nearest', 'mean', 'median', 'bfill', 'ffill', 'random'
        ], default_value='drift')

        cs = ConfigurationSpace()
        cs.add_hyperparameters([method])
        return cs
