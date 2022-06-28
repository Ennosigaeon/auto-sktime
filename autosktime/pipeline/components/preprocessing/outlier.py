from typing import Union

import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeTransformer


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
        # TODO ValueError: The `window_length` and the forecasting horizon are incompatible with the length of `y`.
        #  Found `window_length`=99, `max(fh)`=1, but len(y)=92. It is required that the window length plus maximum
        #  forecast horizon is smaller than the length of the time series `y` itself.
        window_length = UniformIntegerHyperparameter('window_length', lower=3, upper=100, default_value=10, log=True)
        n_sigma = UniformFloatHyperparameter('n_sigma', lower=2, upper=5, default_value=3)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([window_length, n_sigma])
        return cs
