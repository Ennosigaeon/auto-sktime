import os
from collections import OrderedDict
from typing import Dict, Type, List

import pandas as pd

from ConfigSpace import ConfigurationSpace
from autosktime.constants import SUPPORTED_INDEX_TYPES, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, find_components, AutoSktimeChoice
from autosktime.pipeline.components.nn.network.base import BaseNetwork
from autosktime.pipeline.components.nn.util import NN_DATA
from autosktime.pipeline.util import Int64Index

_nn_directory = os.path.split(__file__)[0]
_nns = find_components(__package__, _nn_directory, BaseNetwork)


class NeuralNetworkChoice(AutoSktimeChoice, AutoSktimeComponent):
    _estimator_class: Type[BaseNetwork] = None
    estimator: BaseNetwork = None

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
