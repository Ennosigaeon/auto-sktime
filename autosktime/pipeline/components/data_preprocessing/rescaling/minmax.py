import numpy as np
import pandas as pd
from typing import Tuple

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimePreprocessingAlgorithm, COMPONENT_PROPERTIES
from autosktime.pipeline.util import Int64Index


class MinMaxScalerComponent(AutoSktimePreprocessingAlgorithm):
    def __init__(self, feature_range: Tuple = (0, 1), random_state: np.random.RandomState = None):
        super().__init__()
        from sklearn.preprocessing import MinMaxScaler
        self.estimator = MinMaxScaler(feature_range=feature_range, copy=False)
        self.feature_range = feature_range
        self.random_state = random_state

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }
