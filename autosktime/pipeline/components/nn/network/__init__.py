import os
from collections import OrderedDict
from typing import Dict, Type, List

import pandas as pd
from torch import nn

from ConfigSpace import ConfigurationSpace
from autosktime.constants import SUPPORTED_INDEX_TYPES, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, find_components, AutoSktimeChoice, \
    AutoSktimeRegressionAlgorithm
from autosktime.pipeline.components.nn.util import NN_DATA
from autosktime.pipeline.util import Int64Index

_nn_directory = os.path.split(__file__)[0]
_nns = find_components(__package__, _nn_directory, nn.Module)


class NeuralNetworkChoice(AutoSktimeChoice, AutoSktimeRegressionAlgorithm):
    _estimator_class: Type[nn.Module] = None
    estimator: nn.Module = None

    def get_hyperparameter_search_space(
            self,
            dataset_properties: DatasetProperties = None,
            default: str = 'rnn',
            include: List[str] = None,
            exclude: List[str] = None
    ) -> ConfigurationSpace:
        return super().get_hyperparameter_search_space(dataset_properties, default, include, exclude)

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None):
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }

    @classmethod
    def get_components(cls) -> Dict[str, Type[AutoSktimeComponent]]:
        components = OrderedDict()
        components.update(_nns)
        return components

    def transform(self, X: NN_DATA):
        return self.estimator.transform(X)

    def fit(self, X: NN_DATA, y: pd.Series):
        # noinspection PyUnresolvedReferences
        return self.estimator.fit(X, y)

    def update(self, X: NN_DATA, y: pd.Series, update_params: bool = True):
        # noinspection PyUnresolvedReferences
        self.estimator.update(X, y, n_iter=self.estimator.desired_iterations)
        return self
