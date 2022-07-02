# import warnings
#
# import numpy as np
# import pandas as pd
# from ConfigSpace.conditions import EqualsCondition
# from ConfigSpace.configuration_space import ConfigurationSpace
# from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter
#
# from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
#     HANDLES_PANEL
# from autosktime.data import DatasetProperties
# from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimePreprocessingAlgorithm
# from autosktime.util.common import check_for_bool, check_none
#
#
# class FastICAComponent(AutoSktimePreprocessingAlgorithm):
#     def __init__(
#             self,
#             algorithm: str = 'parallel',
#             whiten: bool = False,
#             fun: str = 'logcosh',
#             n_components: int = None,
#             random_state: np.random.RandomState = None
#     ):
#         super().__init__()
#         self.algorithm = algorithm
#         self.whiten = whiten
#         self.fun = fun
#         self.n_components = n_components
#
#         self.random_state = random_state
#
#     def fit(self, X: pd.DataFrame, y: pd.Series = None):
#         from sklearn.decomposition import FastICA
#
#         self.whiten = check_for_bool(self.whiten)
#         if check_none(self.n_components):
#             self.n_components = None
#         else:
#             self.n_components = int(self.n_components)
#
#         self.estimator = FastICA(
#             n_components=self.n_components, algorithm=self.algorithm,
#             fun=self.fun, whiten=self.whiten, random_state=self.random_state
#         )
#         # Make the RuntimeWarning an Exception!
#         with warnings.catch_warnings():
#             warnings.filterwarnings('error', message='array must not contain infs or NaNs')
#             try:
#                 # noinspection PyUnresolvedReferences
#                 self.estimator.fit(X)
#             except ValueError as e:
#                 if 'array must not contain infs or NaNs' in e.args[0]:
#                     raise ValueError('Bug in scikit-learn: https://github.com/scikit-learn/scikit-learn/pull/2738')
#                 else:
#                     raise
#
#         return self
#
#     @staticmethod
#     def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
#         return {
#             HANDLES_UNIVARIATE: True,
#             HANDLES_MULTIVARIATE: True,
#             HANDLES_PANEL: True,
#             IGNORES_EXOGENOUS_X: False,
#             SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, pd.core.indexes.numeric.Int64Index]
#         }
#
#     @staticmethod
#     def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
#         cs = ConfigurationSpace()
#
#         n_components = UniformIntegerHyperparameter('n_components', 10, 2000, default_value=100)
#         algorithm = CategoricalHyperparameter('algorithm', ['parallel', 'deflation'])
#         whiten = CategoricalHyperparameter('whiten', ['False', 'True'])
#         fun = CategoricalHyperparameter('fun', ['logcosh', 'exp', 'cube'])
#         cs.add_hyperparameters([n_components, algorithm, whiten, fun])
#
#         cs.add_condition(EqualsCondition(n_components, whiten, 'True'))
#
#         return cs
