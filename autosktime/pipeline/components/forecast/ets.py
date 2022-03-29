from typing import Optional

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from sktime.forecasting.base import ForecastingHorizon

from autosktime.pipeline.components.base import AutoSktimeComponent, AutoSktimePredictor


class ETSComponent(AutoSktimeComponent, AutoSktimePredictor):

    def __init__(self, error: str, trend: str, season: str, damped: bool, **kwargs):
        self.error = error
        self.trend = trend
        self.season = season
        self.damped = damped
        self.kwargs = kwargs

    def fit(self, y, X=None, fh: Optional[ForecastingHorizon] = None):
        from sktime.forecasting.ets import AutoETS

        trend = None if self.trend == 'None' else self.trend
        season = None if self.season == 'None' else self.season

        self.estimator = AutoETS(
            error=self.error,
            trend=trend,
            seasonal=season,
            damped_trend=self.damped,
            **self.kwargs
        )
        self.estimator.fit(y, X=X, fh=fh)
        return self

    @staticmethod
    def get_properties(dataset_properties=None):
        from sktime.forecasting.ets import AutoETS
        return AutoETS.get_class_tags()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        cs.add_hyperparameter(CategoricalHyperparameter('error', ['add', 'mul']))
        cs.add_hyperparameter(CategoricalHyperparameter('trend', ['None', 'add', 'mul']))
        cs.add_hyperparameter(CategoricalHyperparameter('season', ['None', 'add', 'mul']))
        cs.add_hyperparameter(CategoricalHyperparameter('damped', [False, True]))

        return cs
