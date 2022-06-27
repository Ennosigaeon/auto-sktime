from typing import Optional, Union

import numpy as np
import pandas as pd

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimePreprocessingAlgorithm


class VarianceThresholdComponent(AutoSktimePreprocessingAlgorithm):
    def __init__(self, random_state: Optional[Union[int, np.random.RandomState]] = None) -> None:
        from sklearn.feature_selection import VarianceThreshold
        super().__init__()
        self.estimator = VarianceThreshold(threshold=0.0)
        self.random_state = random_state

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: False,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }
