import pandas as pd

from autosktime.data import DatasetProperties
from sktime.forecasting.base import ForecastingHorizon

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, ForbiddenInClause, ForbiddenAndConjunction, \
    ForbiddenEqualsClause, InCondition
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MISSING, HANDLES_MULTIVARIATE, \
    SUPPORTED_INDEX_TYPES
from autosktime.pipeline.components.base import AutoSktimePredictor, COMPONENT_PROPERTIES


class ETSComponent(AutoSktimePredictor):

    def __init__(
            self,
            sp: int = 1,
            error: str = 'add',
            trend: str = None,
            seasonal: str = None,
            damped_trend: bool = False,
            random_state=None
    ):
        self.sp = sp
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend

    def fit(self, y, X: pd.DataFrame = None, fh: ForecastingHorizon = None):
        from sktime.forecasting.ets import AutoETS

        trend = None if self.trend == 'None' else self.trend
        seasonal = None if self.seasonal == 'None' else self.seasonal

        self.estimator = AutoETS(
            sp=self.sp,
            error=self.error,
            trend=trend,
            seasonal=seasonal,
            damped_trend=self.damped_trend,
        )
        self.estimator.fit(y, X=X, fh=fh)
        return self

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: False,
            IGNORES_EXOGENOUS_X: True,
            HANDLES_MISSING: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        error = CategoricalHyperparameter('error', ['add', 'mul'])
        trend = CategoricalHyperparameter('trend', ['None', 'add', 'mul'])
        seasonal = CategoricalHyperparameter('seasonal', ['None', 'add', 'mul'])

        damped_trend = CategoricalHyperparameter('damped_trend', [False, True])
        damped_trend_depends_on_trend = InCondition(damped_trend, trend, ['add', 'mul'])

        sp = CategoricalHyperparameter('sp', [1, 2, 4, 7, 12])
        seasonal_sp = ForbiddenAndConjunction(
            ForbiddenInClause(seasonal, ['add', 'mul']),
            ForbiddenEqualsClause(sp, 1)
        )

        cs.add_hyperparameters([error, trend, seasonal, damped_trend, sp])
        cs.add_conditions([damped_trend_depends_on_trend])
        cs.add_forbidden_clauses([seasonal_sp])

        return cs
