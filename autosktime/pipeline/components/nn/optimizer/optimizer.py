import numpy as np
import pandas as pd
from abc import ABC
from torch.optim import Optimizer, Adam
from typing import Optional, Any

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, ForbiddenGreaterThanRelation
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, COMPONENT_PROPERTIES
from autosktime.pipeline.components.nn.util import NN_DATA
from autosktime.pipeline.util import Int64Index


class BaseOptimizerComponent(AutoSktimeComponent, ABC):

    def __init__(self) -> None:
        super().__init__()
        self.optimizer: Optional[Optimizer] = None

    def transform(self, X: NN_DATA) -> NN_DATA:
        X.update({'optimizer': self.optimizer})
        return X

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }


class AdamOptimizer(BaseOptimizerComponent):
    def __init__(
            self,
            lr: float = 0.001,
            beta1: float = 0.9,
            beta2: float = 0.999,
            weight_decay: float = 0.0001,
            random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X: NN_DATA, y: Any = None) -> BaseOptimizerComponent:
        self.optimizer = Adam(
            params=X['network'].parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )
        return self

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        lr = UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-1, default_value=1e-2, log=True)
        beta1 = UniformFloatHyperparameter('beta1', lower=0.85, upper=0.999, default_value=0.9)
        beta2 = UniformFloatHyperparameter('beta2', lower=0.9, upper=0.9999, default_value=0.999)
        weight_decay = UniformFloatHyperparameter('weight_decay', lower=0.0, upper=0.1, default_value=0.0)

        beta1_must_be_smaller_than_beta2 = ForbiddenGreaterThanRelation(beta1, beta2)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([lr, beta1, beta2, weight_decay])
        cs.add_forbidden_clauses([beta1_must_be_smaller_than_beta2])

        return cs
