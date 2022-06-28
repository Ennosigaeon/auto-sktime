from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from ConfigSpace import Configuration
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES
from autosktime.pipeline.components.data_preprocessing.reduction import ReductionChoice
from autosktime.pipeline.components.data_preprocessing.rescaling import RescalingChoice
from autosktime.pipeline.components.data_preprocessing.variance_threshold import VarianceThresholdComponent
from autosktime.pipeline.templates.base import ConfigurablePipeline


class DataPreprocessingPipeline(Pipeline, ConfigurablePipeline):

    def __init__(
            self,
            config: Configuration = None,
            dataset_properties: DatasetProperties = None,
            include: Dict[str, List[str]] = None,
            exclude: Dict[str, List[str]] = None,
            random_state: np.random.RandomState = None,
            init_params: Dict[str, Any] = None
    ):
        self._init(config, dataset_properties, include, exclude, random_state, init_params)
        super().__init__(self.steps)

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, pd.core.indexes.numeric.Int64Index]
        }

    def _get_pipeline_steps(self, dataset_properties: DatasetProperties = None) -> List[Tuple[str, BaseEstimator]]:
        steps = []

        default_dataset_properties = {}
        if dataset_properties is not None and isinstance(dataset_properties, dict):
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ('variance_threshold', VarianceThresholdComponent()),
            ('selection', ReductionChoice()),
            ('rescaling', RescalingChoice()),
        ])

        return steps
