from typing import Tuple

import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.validation._dependencies import _check_soft_dependencies

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, EqualsCondition
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimePredictor, COMPONENT_PROPERTIES


class BATSForecasterComponent(AutoSktimePredictor):
    from sktime.forecasting.bats import BATS

    _estimator_class = BATS

    def __init__(
            self,
            use_box_cox: bool = None,
            box_cox_bounds: Tuple[float, float] = (0, 1),
            use_trend: bool = None,
            use_damped_trend: bool = None,
            sp: int = None,
            use_arma_errors: bool = True,
            random_state=None
    ):
        super().__init__()
        self.use_box_cox = use_box_cox
        self.box_cox_bounds = box_cox_bounds
        self.use_trend = use_trend
        self.use_damped_trend = use_damped_trend
        self.sp = sp
        self.use_arma_errors = use_arma_errors
        self.random_state = random_state

    def _fit(self, y, X: pd.DataFrame = None, fh: ForecastingHorizon = None):
        self.estimator = self._estimator_class(
            use_box_cox=self.use_box_cox,
            box_cox_bounds=self.box_cox_bounds,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            sp=[self.sp],
            use_arma_errors=self.use_arma_errors,
            n_jobs=1
        )
        self.estimator.fit(y, X=X, fh=fh)
        return self

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            IGNORES_EXOGENOUS_X: True,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        use_box_cox = CategoricalHyperparameter('use_box_cox', [False, True])
        # TODO
        # box_cox_bounds = UniformFloatHyperparameter('box_cox_bounds', [False, True])

        use_trend = CategoricalHyperparameter('use_trend', [False, True])
        use_damped_trend = CategoricalHyperparameter('use_damped_trend', [False, True])
        use_damped_trend_depends_on_trend = EqualsCondition(use_damped_trend, use_trend, True)

        # TODO BATS supports multiple seasonal periods
        sp = CategoricalHyperparameter('sp', [1, 2, 4, 7, 12])

        use_arma_errors = CategoricalHyperparameter('use_arma_errors', [False, True], default_value=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([use_box_cox, use_trend, use_damped_trend, sp, use_arma_errors])
        cs.add_conditions([use_damped_trend_depends_on_trend])
        return cs

    @staticmethod
    def check_dependencies():
        _check_soft_dependencies('tbats', severity='error')
