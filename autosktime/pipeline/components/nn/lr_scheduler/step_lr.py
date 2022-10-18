import pandas as pd
from torch.optim.lr_scheduler import StepLR
from typing import List, Any

from ConfigSpace import ConfigurationSpace
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, COMPONENT_PROPERTIES
from autosktime.pipeline.components.nn.util import NN_DATA
from autosktime.pipeline.util import Int64Index


class LearningRateScheduler(AutoSktimeComponent):

    def __init__(self):
        super().__init__()
        self.scheduler = None

    def fit(self, X: NN_DATA, y: Any = None) -> AutoSktimeComponent:
        optimizer = X['optimizer']
        self.scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
        return self

    def transform(self, X: NN_DATA) -> NN_DATA:
        X.update({'scheduler': self.scheduler})
        return X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        return ConfigurationSpace()

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }

    def state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        return self.scheduler.load_state_dict(state_dict)

    def get_last_lr(self) -> List[float]:
        return self.scheduler.get_last_lr()

    def get_lr(self) -> float:
        return self.scheduler.get_lr()

    def step(self, *args, **kwargs) -> None:
        return self.scheduler.step(*args, **kwargs)
