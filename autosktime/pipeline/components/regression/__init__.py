import os
from collections import OrderedDict
from typing import Dict, Type

import pandas as pd
from autosktime.constants import SUPPORTED_INDEX_TYPES, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, find_components, AutoSktimeChoice, \
    AutoSktimeRegressionAlgorithm
from sklearn.base import RegressorMixin

_regressor_directory = os.path.split(__file__)[0]
_regressors = find_components(__package__, _regressor_directory, AutoSktimeRegressionAlgorithm)


class RegressorChoice(AutoSktimeChoice, AutoSktimeRegressionAlgorithm):
    _estimator_class: Type[RegressorMixin] = None
    estimator: RegressorMixin = None

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None):
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: False,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }

    @classmethod
    def get_components(cls) -> Dict[str, Type[AutoSktimeComponent]]:
        components = OrderedDict()
        components.update(_regressors)
        return components

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # noinspection PyUnresolvedReferences
        return self.estimator.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # noinspection PyUnresolvedReferences
        return self.estimator.predict(X)
