import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, Constant, \
    UniformFloatHyperparameter
from sktime.forecasting.base import ForecastingHorizon
# noinspection PyProtectedMember
from sktime.utils.validation._dependencies import _check_soft_dependencies

from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimePredictor, COMPONENT_PROPERTIES


class ProphetComponent(AutoSktimePredictor):
    from sktime.forecasting.fbprophet import Prophet

    _estimator_class = Prophet

    def __init__(
            self,
            freq: str = None,
            growth: str = 'linear',
            n_changepoints: int = 25,
            changepoint_prior_scale: float = 0.05,
            seasonality_prior_scale: float = 10,
            seasonality_mode: str = 'additive',
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.freq = freq
        self.growth = growth
        self.n_changepoints = n_changepoints
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.random_state = random_state

    def _fit(self, y, X: pd.DataFrame = None, fh: ForecastingHorizon = None):
        self.estimator = self._estimator_class(
            freq=self.freq,
            growth=self.growth,
            n_changepoints=self.n_changepoints,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            verbose=False
        )

        self.estimator.fit(y, X=X, fh=fh)
        return self

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.DatetimeIndex]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        # TODO capacity is required for logistic growth, see
        #  https://facebook.github.io/prophet/docs/saturating_forecasts.html
        growth = Constant('growth', 'linear')
        freq = Constant('freq', dataset_properties.frequency)

        changepoint_prior_scale = UniformFloatHyperparameter('changepoint_prior_scale', 0.001, 0.5, default_value=0.05)
        seasonality_prior_scale = UniformFloatHyperparameter('seasonality_prior_scale', 0.001, 10, default_value=10)
        changepoint_range = UniformFloatHyperparameter('changepoint_range', 0.8, 0.95, default_value=0.8)

        n_changepoints = UniformIntegerHyperparameter('n_changepoints', 0, 50, default_value=25)

        seasonality_mode = CategoricalHyperparameter('seasonality_mode', ['additive', 'multiplicative'])

        cs = ConfigurationSpace()
        cs.add_hyperparameters([
            freq, growth, n_changepoints, changepoint_prior_scale, seasonality_prior_scale, changepoint_range,
            seasonality_mode
        ])

        return cs

    @staticmethod
    def check_dependencies():
        _check_soft_dependencies('prophet', severity='error')
