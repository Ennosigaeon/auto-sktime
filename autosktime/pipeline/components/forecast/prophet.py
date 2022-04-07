from typing import Union

import pandas as pd

from autosktime.data import DatasetProperties
from sktime.forecasting.base import ForecastingHorizon

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, Constant
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MISSING, HANDLES_MULTIVARIATE, \
    SUPPORTED_INDEX_TYPES
from autosktime.pipeline.components.base import AutoSktimePredictor, COMPONENT_PROPERTIES


class ProphetComponent(AutoSktimePredictor):

    def __init__(
            self,
            growth: str = 'linear',
            n_changepoints: int = 25,
            yearly_seasonality: Union[bool, str] = 'auto',
            weekly_seasonality: Union[bool, str] = 'auto',
            daily_seasonality: Union[bool, str] = 'auto',
            seasonality_mode: str = 'additive',
            random_state=None
    ):
        self.growth = growth
        self.n_changepoints = n_changepoints
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode

    def fit(self, y, X: pd.DataFrame = None, fh: ForecastingHorizon = None):
        from sktime.forecasting.fbprophet import Prophet

        def as_bool(value: Union[str, bool]) -> bool:
            try:
                return bool(value)
            except ValueError:
                return value

        yearly_seasonality = as_bool(self.yearly_seasonality)
        weekly_seasonality = as_bool(self.weekly_seasonality)
        daily_seasonality = as_bool(self.daily_seasonality)

        self.estimator = Prophet(
            growth=self.growth,
            n_changepoints=self.n_changepoints,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            verbose=False
        )

        self.estimator.fit(y, X=X, fh=fh)
        return self

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: False,
            IGNORES_EXOGENOUS_X: False,
            HANDLES_MISSING: False,
            SUPPORTED_INDEX_TYPES: [pd.DatetimeIndex]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        # TODO capacity is required for logistic growth, see
        #  https://facebook.github.io/prophet/docs/saturating_forecasts.html
        # growth = CategoricalHyperparameter('growth', ['linear', 'logistic'])
        growth = Constant('growth', 'linear')
        n_changepoints = UniformIntegerHyperparameter('n_changepoints', 0, 50, default_value=25)

        yearly_seasonality = CategoricalHyperparameter('yearly_seasonality', ['auto', 'True', 'False'])
        weekly_seasonality = CategoricalHyperparameter('weekly_seasonality', ['auto', 'True', 'False'])
        daily_seasonality = CategoricalHyperparameter('daily_seasonality', ['auto', 'True', 'False'])

        seasonality_mode = CategoricalHyperparameter('seasonality_mode', ['additive', 'multiplicative'])

        cs = ConfigurationSpace()
        cs.add_hyperparameters([
            growth, n_changepoints, yearly_seasonality, weekly_seasonality, daily_seasonality, seasonality_mode
        ])
        cs.add_conditions([])
        cs.add_forbidden_clauses([])

        return cs
