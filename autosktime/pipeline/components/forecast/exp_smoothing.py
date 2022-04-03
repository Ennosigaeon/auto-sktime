from typing import Optional

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, ForbiddenInClause, ForbiddenAndConjunction, \
    ForbiddenEqualsClause, InCondition
from sktime.forecasting.base import ForecastingHorizon

from autosktime.pipeline.components.base import AutoSktimePredictor, DATASET_PROPERTIES, COMPONENT_PROPERTIES


class ExponentialSmoothingComponent(AutoSktimePredictor):

    def __init__(
            self,
            sp: int = 1,
            trend: str = None,
            seasonal: str = None,
            damped_trend: bool = False,
            random_state=None
    ):
        self.sp = sp
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend

    def fit(self, y, X=None, fh: Optional[ForecastingHorizon] = None):
        from sktime.forecasting.exp_smoothing import ExponentialSmoothing

        trend = None if self.trend == 'None' else self.trend
        seasonal = None if self.seasonal == 'None' else self.seasonal

        self.estimator = ExponentialSmoothing(
            sp=self.sp,
            trend=trend,
            seasonal=seasonal,
            damped_trend=self.damped_trend,
        )
        self.estimator.fit(y, X=X, fh=fh)
        return self

    @staticmethod
    def get_properties(dataset_properties: DATASET_PROPERTIES = None) -> COMPONENT_PROPERTIES:
        from sktime.forecasting.exp_smoothing import ExponentialSmoothing
        return ExponentialSmoothing.get_class_tags()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DATASET_PROPERTIES = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        trend = CategoricalHyperparameter('trend', ['None', 'add', 'mul'])
        seasonal = CategoricalHyperparameter('seasonal', ['None', 'add', 'mul'])

        damped_trend = CategoricalHyperparameter('damped_trend', [False, True])
        damped_trend_depends_on_trend = InCondition(damped_trend, trend, ['add', 'mul'])

        sp = CategoricalHyperparameter('sp', [1, 2, 4, 7, 12])
        seasonal_sp = ForbiddenAndConjunction(
            ForbiddenInClause(seasonal, ['add', 'mul']),
            ForbiddenEqualsClause(sp, 1)
        )

        cs.add_hyperparameters([trend, seasonal, damped_trend, sp])
        cs.add_conditions([damped_trend_depends_on_trend])
        cs.add_forbidden_clauses([seasonal_sp])

        return cs
