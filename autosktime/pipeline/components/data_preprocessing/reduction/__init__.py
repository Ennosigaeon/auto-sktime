import os
from collections import OrderedDict

import pandas as pd

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeChoice, find_components, AutoSktimePreprocessingAlgorithm, \
    COMPONENT_PROPERTIES

classifier_directory = os.path.split(__file__)[0]
_preprocessors = find_components(__package__, classifier_directory, AutoSktimePreprocessingAlgorithm)


class ReductionChoice(AutoSktimeChoice, AutoSktimePreprocessingAlgorithm):

    @classmethod
    def get_components(cls):
        components = OrderedDict()
        components.update(_preprocessors)
        return components

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }
