from typing import Union

import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeTransformer, \
    AutoSktimePreprocessingAlgorithm
from autosktime.pipeline.util import Int64Index


class IdentityComponent(AutoSktimeTransformer):
    _tags = {
        'capability:inverse_transform': True
    }

    def __init__(self, random_state: np.random.RandomState = None):
        super().__init__()
        self.random_state = random_state

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        return self

    def _transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        return X

    def _inverse_transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        return X

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
        cs = ConfigurationSpace()
        return cs


class IdentityPreprocessing(AutoSktimePreprocessingAlgorithm):

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # noinspection PyTypeChecker
        self.estimator = 'passthrough'
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }
