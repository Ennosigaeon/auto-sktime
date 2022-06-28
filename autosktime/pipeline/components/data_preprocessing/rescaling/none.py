import pandas as pd

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimePreprocessingAlgorithm


class NoRescalingComponent(AutoSktimePreprocessingAlgorithm):

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
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, pd.core.indexes.numeric.Int64Index]
        }
