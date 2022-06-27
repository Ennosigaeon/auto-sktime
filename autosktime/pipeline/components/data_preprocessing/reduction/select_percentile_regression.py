from functools import partial

import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimePreprocessingAlgorithm, COMPONENT_PROPERTIES


class SelectPercentileRegression(AutoSktimePreprocessingAlgorithm):

    def __init__(self, percentile: int = 50, score_func: str = 'f_regression', random_state=None):
        super().__init__()
        import sklearn.feature_selection

        self.random_state = random_state  # We don't use this
        self.percentile = int(float(percentile))
        if score_func == 'f_regression':
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == 'mutual_info':
            self.score_func = partial(sklearn.feature_selection.mutual_info_regression,
                                      random_state=self.random_state)
        else:
            raise ValueError(f'Unknown score function {score_func}')

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import sklearn.feature_selection

        self.estimator = sklearn.feature_selection.SelectPercentile(
            score_func=self.score_func,
            percentile=self.percentile)

        # noinspection PyUnresolvedReferences
        self.estimator.fit(X, y)
        return self

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        percentile = UniformFloatHyperparameter('percentile', lower=1, upper=99, default_value=50)
        score_func = CategoricalHyperparameter('score_func', choices=['f_regression', 'mutual_info'])

        cs = ConfigurationSpace()
        cs.add_hyperparameters([percentile, score_func])
        return cs
