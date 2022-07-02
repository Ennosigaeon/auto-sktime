from typing import Tuple, List

import pandas as pd
from sktime.forecasting.compose._pipeline import SUPPORTED_MTYPES

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, COMPONENT_PROPERTIES
from autosktime.pipeline.components.preprocessing.detrend import DetrendComponent
from autosktime.pipeline.components.preprocessing.impute import ImputerComponent
from autosktime.pipeline.components.preprocessing.outlier import HampelFilterComponent
from autosktime.pipeline.components.reduction.reduction import ReductionComponent
from autosktime.pipeline.templates.base import ConfigurableTransformedTargetForecaster
from autosktime.pipeline.util import Int64Index


class RegressionPipeline(ConfigurableTransformedTargetForecaster):
    _tags = {
        "scitype:y": "both",
        "y_inner_mtype": SUPPORTED_MTYPES,
        "X_inner_mtype": SUPPORTED_MTYPES,
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": True,
        'X-y-must-have-same-index': True
    }

    def _get_pipeline_steps(self) -> List[Tuple[str, AutoSktimeComponent]]:
        steps = [
            ('detrend', DetrendComponent(random_state=self.random_state)),
            ('outlier', HampelFilterComponent(random_state=self.random_state)),
            ('imputation', ImputerComponent(random_state=self.random_state)),
            ('reduction', ReductionComponent(random_state=self.random_state))
        ]
        return steps

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            # sktime reduction can not handle multi-index as of 0.12.x
            HANDLES_PANEL: False,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }
