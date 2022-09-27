from typing import Any, Dict

import pandas as pd
import torch

from ConfigSpace import ConfigurationSpace
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, COMPONENT_PROPERTIES
from autosktime.pipeline.util import Int64Index

NN_DATA = Dict[str, Any]


class DictionaryInput(AutoSktimeComponent):

    def fit_transform(self, X, y):
        return {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'X': X,
            'y': y
        }

    def transform(self, X, y=None):
        return {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'X': X,
            'y': y
        }

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        return ConfigurationSpace()


def noop(x=None, *args, **kwargs):
    "Do nothing"
    return x
