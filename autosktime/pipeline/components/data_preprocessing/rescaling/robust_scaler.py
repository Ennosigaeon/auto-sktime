import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimePreprocessingAlgorithm
from autosktime.pipeline.util import Int64Index


class RobustScalerComponent(AutoSktimePreprocessingAlgorithm):
    def __init__(
            self,
            q_min: float = 0.25,
            q_max: float = 0.75,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        from sklearn.preprocessing import RobustScaler
        self.q_min = q_min
        self.q_max = q_max
        self.estimator = RobustScaler(quantile_range=(self.q_min, self.q_max), copy=False)
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

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        q_min = UniformFloatHyperparameter('q_min', 0.001, 0.3, default_value=0.25)
        q_max = UniformFloatHyperparameter('q_max', 0.7, 0.999, default_value=0.75)
        cs.add_hyperparameters((q_min, q_max))
        return cs
