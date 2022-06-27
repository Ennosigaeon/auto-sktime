from functools import partial

import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter

from ConfigSpace import NotEqualsCondition
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimePreprocessingAlgorithm, COMPONENT_PROPERTIES


class SelectRegressionRates(AutoSktimePreprocessingAlgorithm):
    def __init__(self, alpha, mode='percentile', score_func='f_regression', random_state=None):
        super().__init__()
        import sklearn.feature_selection

        self.random_state = random_state  # We don't use this
        self.alpha = alpha
        self.mode = mode

        if score_func == 'f_regression':
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == 'mutual_info_regression':
            self.score_func = partial(sklearn.feature_selection.mutual_info_regression, random_state=self.random_state)
            # Mutual info consistently crashes if percentile is not the mode
            self.mode = 'percentile'
        else:
            raise ValueError(f'Unknown score function {score_func}')

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        import sklearn.feature_selection

        self.alpha = float(self.alpha)

        self.estimator = sklearn.feature_selection.GenericUnivariateSelect(score_func=self.score_func, param=self.alpha,
                                                                           mode=self.mode)

        # noinspection PyUnresolvedReferences
        self.estimator.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.estimator is None:
            raise NotImplementedError()

        try:
            # noinspection PyUnresolvedReferences
            Xt = self.estimator.transform(X)
        except ValueError as e:
            if 'zero-size array to reduction operation maximum which has no identity' in e.message:
                raise ValueError(f'{self.__class__.__name__} removed all features.')
            else:
                raise e

        if Xt.shape[1] == 0:
            raise ValueError(f'{self.__class__.__name__} removed all features.')
        return Xt

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: False,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        alpha = UniformFloatHyperparameter('alpha', lower=0.01, upper=0.5, default_value=0.1)

        if dataset_properties is not None and dataset_properties.get('sparse'):
            choices = ['f_regression', 'mutual_info_regression']
        else:
            choices = ['f_regression']

        score_func = CategoricalHyperparameter('score_func', choices=choices)

        mode = CategoricalHyperparameter('mode', ['fpr', 'fdr', 'fwe'])

        cs = ConfigurationSpace()
        cs.add_hyperparameter(alpha)
        cs.add_hyperparameter(score_func)
        cs.add_hyperparameter(mode)

        # Mutual info consistently crashes if percentile is not the mode
        if 'mutual_info_regression' in choices:
            cond = NotEqualsCondition(mode, score_func, 'mutual_info_regression')
            cs.add_condition(cond)

        return cs
