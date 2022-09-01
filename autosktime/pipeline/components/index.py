import numpy as np
import pandas as pd

from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimePreprocessingAlgorithm
from autosktime.pipeline.util import Int64Index


class AddIndexComponent(AutoSktimePreprocessingAlgorithm):

    def __init__(self, random_state: np.random.RandomState = None):
        super().__init__()
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        index = X.index
        if isinstance(index, pd.MultiIndex):
            index = index.droplevel(0)

        Xt = X.copy()
        Xt['__index__'] = index

        return Xt

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: True,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }
