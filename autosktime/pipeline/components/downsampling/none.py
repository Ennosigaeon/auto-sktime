from typing import Union

import numpy as np
import pandas as pd
from sktime.base._base import TagAliaserMixin

from ConfigSpace import ConfigurationSpace
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES
from autosktime.pipeline.components.downsampling import BaseDownSampling
from autosktime.pipeline.util import Int64Index


class IdentityComponent(BaseDownSampling):
    _tags = {
        'capability:inverse_transform': True
    }

    def __init__(self, random_state: np.random.RandomState = None):
        super().__init__()
        self.random_state = random_state

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        return self

    def _transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        return X, y

    def _inverse_transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        return X, y

    def get_tags(self):
        tags = super(TagAliaserMixin, self).get_tags()
        return tags

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
