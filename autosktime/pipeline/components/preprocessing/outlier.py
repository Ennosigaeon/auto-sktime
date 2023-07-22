from typing import Union

import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeTransformer
from autosktime.pipeline.components.preprocessing.impute import ImputerComponent
from autosktime.pipeline.util import Int64Index


class HampelFilterComponent(AutoSktimeTransformer):
    from sktime.transformations.series.outlier_detection import HampelFilter

    _estimator_class = HampelFilter

    def __init__(self, window_length: int = 10, n_sigma: float = 3., random_state: np.random.RandomState = None):
        super().__init__()
        self.window_length = window_length
        self.n_sigma = n_sigma
        self.random_state = random_state

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        self.estimator = self._estimator_class(window_length=self.window_length, n_sigma=self.n_sigma)
        self.estimator.fit(X=X, y=y)
        self.imputer = ImputerComponent(method='ffill', random_state=self.random_state)
        self.imputer.fit(X=X, y=y)
        return self

    def _transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        if self.estimator is None:
            raise NotImplementedError

        # hampel filter implementation uses integers to access index
        index = X.index
        X.index = pd.RangeIndex(start=0, stop=len(index))
        res = self.estimator.transform(X, y=y)
        res.index = index
        res = self.imputer.transform(res, y=y)
        return res

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
        window_length = UniformIntegerHyperparameter('window_length', lower=3, log=True,
                                                     upper=min(dataset_properties.series_length - 1, 100),
                                                     default_value=min(dataset_properties.series_length - 1, 5))
        n_sigma = UniformFloatHyperparameter('n_sigma', lower=2, upper=5, default_value=3)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([window_length, n_sigma])
        return cs
