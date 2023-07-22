from itertools import product
from typing import List, Tuple

import pandas as pd
from sktime.forecasting.compose._pipeline import SUPPORTED_MTYPES

from ConfigSpace import ConfigurationSpace, ForbiddenAndConjunction, ForbiddenEqualsClause
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, COMPONENT_PROPERTIES
from autosktime.pipeline.components.forecast import ForecasterChoice
from autosktime.pipeline.components.normalizer import NormalizerChoice
from autosktime.pipeline.components.preprocessing.impute import ImputerComponent
from autosktime.pipeline.components.preprocessing.outlier import HampelFilterComponent
from autosktime.pipeline.components.preprocessing.shift import ShiftTransformerComponent
from autosktime.pipeline.templates.base import ConfigurableTransformedTargetForecaster
from autosktime.pipeline.util import Int64Index


class UnivariateEndogenousPipeline(ConfigurableTransformedTargetForecaster):
    _tags = {
        "scitype:y": "both",
        "y_inner_mtype": SUPPORTED_MTYPES,
        "X_inner_mtype": SUPPORTED_MTYPES,
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
        "capability:pred_int": True,
        'X-y-must-have-same-index': True
    }

    def get_hyperparameter_search_space(self, dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        if not hasattr(self, 'config_space') or self.config_space is None:
            cs = self._get_hyperparameter_search_space()

            forecasters = cs.get_hyperparameter('forecaster:__choice__').choices
            normalizers = cs.get_hyperparameter('normalizer:__choice__').choices

            for f, n in product(['ets', 'theta', 'exp_smoothing'], ['scaled_logit']):
                if f not in forecasters or n not in normalizers:
                    continue
                cs.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(cs.get_hyperparameter('forecaster:__choice__'), f),
                    ForbiddenEqualsClause(cs.get_hyperparameter('normalizer:__choice__'), n))
                )

            self.config_space = cs

        return self.config_space

    def _get_pipeline_steps(self) -> List[Tuple[str, AutoSktimeComponent]]:
        steps = [
            ('imputation', ImputerComponent(random_state=self.random_state)),
            ('shift', ShiftTransformerComponent(random_state=self.random_state)),
            ('outlier', HampelFilterComponent(random_state=self.random_state)),
            ('normalizer', NormalizerChoice(random_state=self.random_state)),
            ('forecaster', ForecasterChoice(random_state=self.random_state))
        ]
        return steps

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: False,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }
