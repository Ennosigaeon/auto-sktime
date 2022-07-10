import os
from collections import OrderedDict
from typing import Dict, Type, List

import pandas as pd

from ConfigSpace import ConfigurationSpace
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import find_components, AutoSktimeChoice, AutoSktimePreprocessingAlgorithm, \
    AutoSktimeComponent, COMPONENT_PROPERTIES
from autosktime.pipeline.util import Int64Index

rescaling_directory = os.path.split(__file__)[0]
_rescalers = find_components(__package__, rescaling_directory, AutoSktimePreprocessingAlgorithm)


class RescalingChoice(AutoSktimeChoice, AutoSktimePreprocessingAlgorithm):

    def get_hyperparameter_search_space(
            self,
            dataset_properties: DatasetProperties = None,
            default: str = 'standardize',
            include: List[str] = None,
            exclude: List[str] = None
    ) -> ConfigurationSpace:
        return super().get_hyperparameter_search_space(dataset_properties, default, include, exclude)

    @classmethod
    def get_components(cls) -> Dict[str, Type[AutoSktimeComponent]]:
        components = OrderedDict()
        components.update(_rescalers)
        return components

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }
