import pandas as pd

from autosktime.data import DatasetProperties
from sktime.forecasting.base import ForecastingHorizon

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MISSING, HANDLES_MULTIVARIATE, \
    SUPPORTED_INDEX_TYPES
from autosktime.pipeline.components.base import AutoSktimePredictor, COMPONENT_PROPERTIES


class ThetaComponent(AutoSktimePredictor):

    def __init__(
            self,
            sp: int = 1,
            deseasonalize: bool = True,
            random_state=None
    ):
        self.sp = sp
        self.deseasonalize = deseasonalize

    def fit(self, y, X: pd.DataFrame = None, fh: ForecastingHorizon = None):
        from sktime.forecasting.theta import ThetaForecaster
        self.estimator = ThetaForecaster(
            sp=self.sp,
            deseasonalize=self.deseasonalize,
        )
        self.estimator.fit(y, X=X, fh=fh)
        return self

    def predict(self, fh: ForecastingHorizon = None, X: pd.DataFrame = None):
        prediction = super().predict(fh, X)

        if self.sp > 1 and fh is not None and self.estimator.fh[0] == fh.to_pandas()[0]:
            # NaiveForecaster uses the last self.sp terms for forecasting. In case that the training
            # data are predicted again, the first self.sp terms are missing and set to nan
            prediction = prediction.backfill()

        return prediction

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: False,
            IGNORES_EXOGENOUS_X: True,
            HANDLES_MISSING: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.PeriodIndex]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        deseasonalize = CategoricalHyperparameter('deseasonalize', [True, False])
        sp = CategoricalHyperparameter('sp', [1, 2, 4, 7, 12])

        cs = ConfigurationSpace()
        cs.add_hyperparameters([deseasonalize, sp])
        return cs
