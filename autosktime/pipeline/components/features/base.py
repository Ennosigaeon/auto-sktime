from abc import ABC

import numpy as np
import pandas as pd

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimePreprocessingAlgorithm, COMPONENT_PROPERTIES
from autosktime.pipeline.util import Int64Index


class BaseFeatureGenerator(AutoSktimePreprocessingAlgorithm, ABC):

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: True,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }
