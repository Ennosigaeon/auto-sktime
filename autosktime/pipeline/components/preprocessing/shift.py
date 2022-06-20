from typing import Union

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace

from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeTransformer, COMPONENT_PROPERTIES
from sktime.transformations.base import BaseTransformer


class ShiftTransformerComponent(AutoSktimeTransformer):
    _tags = {
        'capability:inverse_transform': True
    }

    def __init__(self, lower_bound: float = 10, padding: float = 1.5, random_state=None):
        super().__init__()
        self.lower_bound = lower_bound
        self.padding = padding
        self.random_state = random_state

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        min = X.min()
        if np.any(min < self.lower_bound):
            self.offest_ = np.clip(min - self.lower_bound, -np.inf, 0) * -self.padding
        else:
            self.offest_ = 0

    def _transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        return X + self.offest_

    def _inverse_transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        return X - self.offest_

    def get_tags(self):
        tags = super(BaseTransformer, self).get_tags()
        return tags

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {

        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        return ConfigurationSpace()
