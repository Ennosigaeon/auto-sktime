from typing import Optional, Union

from sktime.forecasting.base import ForecastingHorizon

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, Constant
from autosktime.pipeline.components.base import AutoSktimePredictor, DATASET_PROPERTIES, COMPONENT_PROPERTIES


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

    def fit(self, y, X=None, fh: Optional[ForecastingHorizon] = None):
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
    def get_properties(dataset_properties: DATASET_PROPERTIES = None) -> COMPONENT_PROPERTIES:
        from sktime.forecasting.fbprophet import Prophet
        return Prophet.get_class_tags()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DATASET_PROPERTIES = None) -> ConfigurationSpace:
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
