from typing import Optional

from sktime.forecasting.base import ForecastingHorizon

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from autosktime.pipeline.components.base import AutoSktimePredictor, DATASET_PROPERTIES, COMPONENT_PROPERTIES


class NaiveForecasterComponent(AutoSktimePredictor):

    def __init__(
            self,
            sp: int = 1,
            strategy: str = 'last',
            random_state=None
    ):
        self.sp = sp
        self.strategy = strategy

    def fit(self, y, X=None, fh: Optional[ForecastingHorizon] = None):
        from sktime.forecasting.naive import NaiveForecaster
        self.estimator = NaiveForecaster(
            sp=self.sp,
            strategy=self.strategy,
        )
        self.estimator.fit(y, X=X, fh=fh)
        return self

    def predict(
            self,
            fh: Optional[ForecastingHorizon] = None,
            X=None
    ):
        prediction = super().predict(fh, X)

        if self.sp > 1 and fh is not None and self.estimator.fh[0] == fh.to_pandas()[0]:
            # NaiveForecaster uses the last self.sp terms for forecasting. In case that the training
            # data are predicted again, the first self.sp terms are missing and set to nan
            prediction = prediction.backfill()

        return prediction

    @staticmethod
    def get_properties(dataset_properties: DATASET_PROPERTIES = None) -> COMPONENT_PROPERTIES:
        from sktime.forecasting.naive import NaiveForecaster
        return NaiveForecaster.get_class_tags()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DATASET_PROPERTIES = None) -> ConfigurationSpace:
        strategy = CategoricalHyperparameter('strategy', ['last', 'mean', 'drift'])
        sp = CategoricalHyperparameter('sp', [1, 2, 4, 7, 12])

        cs = ConfigurationSpace()
        cs.add_hyperparameters([strategy, sp])
        return cs
