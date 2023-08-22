from typing import Tuple, List

import pandas as pd
from sktime.forecasting.compose._pipeline import SUPPORTED_MTYPES

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, COMPONENT_PROPERTIES, UpdatablePipeline
from autosktime.pipeline.components.data_preprocessing import VarianceThresholdComponent
from autosktime.pipeline.components.data_preprocessing.rescaling.standardize import StandardScalerComponent
from autosktime.pipeline.components.nn.data_loader import ChunkedDataLoaderComponent
from autosktime.pipeline.components.nn.lr_scheduler import LearningRateScheduler
from autosktime.pipeline.components.nn.network import NeuralNetworkChoice
from autosktime.pipeline.components.nn.optimizer.optimizer import AdamOptimizer
from autosktime.pipeline.components.nn.trainer import TrainerComponent
from autosktime.pipeline.components.nn.util import DictionaryInput
from autosktime.pipeline.components.normalizer.standardize import TargetStandardizeComponent
from autosktime.pipeline.components.preprocessing.impute import ImputerComponent
from autosktime.pipeline.components.reduction.panel import RecursivePanelReducer
from autosktime.pipeline.templates.base import ConfigurableTransformedTargetForecaster
from autosktime.pipeline.util import Int64Index


class NNRegressionPipeline(ConfigurableTransformedTargetForecaster):
    _tags = {
        "scitype:y": "univariate",
        "y_inner_mtype": SUPPORTED_MTYPES,
        "X_inner_mtype": SUPPORTED_MTYPES,
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
        "capability:pred_int": True,
        'X-y-must-have-same-index': True
    }

    def _get_pipeline_steps(self) -> List[Tuple[str, AutoSktimeComponent]]:
        pipeline = UpdatablePipeline(steps=[
            ('variance_threshold', VarianceThresholdComponent(random_state=self.random_state)),
            ('scaling', StandardScalerComponent(random_state=self.random_state)),
            ('dict', DictionaryInput()),
            ('data_loader', ChunkedDataLoaderComponent(random_state=self.random_state)),
            ('network', NeuralNetworkChoice(random_state=self.random_state)),
            ('optimizer', AdamOptimizer(random_state=self.random_state)),
            ('lr_scheduler', LearningRateScheduler()),
            ('trainer', TrainerComponent(random_state=self.random_state)),
        ])

        steps = [
            # Detrending by collapsing panel data to univariate timeseries by averaging
            ('imputation', ImputerComponent(random_state=self.random_state)),
            ('scaling', TargetStandardizeComponent(random_state=self.random_state)),
            ('reduction', RecursivePanelReducer(
                estimator=pipeline,
                random_state=self.random_state,
                dataset_properties=self.dataset_properties)
             ),
        ]
        return steps

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: False,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }

    def supports_pynisher(self) -> bool:
        return False
