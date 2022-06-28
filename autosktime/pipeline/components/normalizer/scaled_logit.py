from typing import Union

import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeTransformer


class ScaledLogitComponent(AutoSktimeTransformer):
    from sktime.transformations.series.scaledlogit import ScaledLogitTransformer

    _estimator_class = ScaledLogitTransformer

    def __init__(
            self,
            lower_bound: float = 0.75,
            upper_bound: float = 1.25,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.random_state = random_state

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        min = X.min()
        max = X.max()

        if min < 0:
            lower_bound = 2 - self.lower_bound
        else:
            lower_bound = self.lower_bound

        if max < 0:
            upper_bound = 2 - self.upper_bound
        else:
            upper_bound = self.upper_bound

        self.estimator = self._estimator_class(lower_bound=lower_bound * min, upper_bound=upper_bound * max)

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
        lower_bound = UniformFloatHyperparameter('lower_bound', 0, 1, default_value=0.75)
        upper_bound = UniformFloatHyperparameter('upper_bound', 1, 2, default_value=1.25)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([lower_bound, upper_bound])
        return cs
