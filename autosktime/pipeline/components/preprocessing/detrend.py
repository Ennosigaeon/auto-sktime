from typing import Union

import numpy as np
import pandas as pd
from sktime.forecasting.trend import PolynomialTrendForecaster

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeTransformer
from autosktime.pipeline.util import Int64Index


class DetrendComponent(AutoSktimeTransformer):
    from sktime.transformations.series.detrend import Detrender

    _estimator_class = Detrender

    def __init__(self, degree: int = 1, with_intercept: bool = True, random_state: np.random.RandomState = None):
        super().__init__()
        self.degree = degree
        self.with_intercept = with_intercept
        self.random_state = random_state

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        self.estimator = self._estimator_class(forecaster=PolynomialTrendForecaster(
            degree=self.degree,
            with_intercept=self.with_intercept
        ))

        self.estimator.fit(X=X, y=y)
        return self

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
        degree = UniformIntegerHyperparameter('degree', lower=1, upper=2, default_value=1)
        with_intercept = CategoricalHyperparameter('with_intercept', [True, False])

        cs = ConfigurationSpace()
        cs.add_hyperparameters([degree, with_intercept])
        return cs
