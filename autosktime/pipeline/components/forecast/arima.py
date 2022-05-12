from typing import Union

import pandas as pd

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, \
    Constant, InCondition, ForbiddenGreaterThanRelation
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimePredictor, COMPONENT_PROPERTIES
from sktime.forecasting.base import ForecastingHorizon


class ARIMAComponent(AutoSktimePredictor):
    from sktime.forecasting.arima import ARIMA

    _estimator_class = ARIMA

    def __init__(
            self,
            p: int = 1,
            d: int = 0,
            q: int = 0,
            P: int = 0,
            D: int = 0,
            Q: int = 0,
            sp: int = 0,
            maxiter: int = 50,
            with_intercept: Union[bool, str] = True,
            random_state=None
    ):
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.sp = sp
        self.maxiter = maxiter
        self.with_intercept = with_intercept
        self.random_state = random_state

    def _fit(self, y, X: pd.DataFrame = None, fh: ForecastingHorizon = None):
        if self.d >= y.shape[0]:
            raise ValueError(f'Trainings data is too short ({y.shape[0]}) for selected d ({self.d}). '
                             f'Try to increase the trainings data or decrease d')

        try:
            with_intercept = bool(self.with_intercept)
        except ValueError:
            with_intercept = self.with_intercept

        self.estimator = self._estimator_class(
            order=(self.p, self.d, self.q),
            seasonal_order=(self.P, self.D, self.Q, self.sp),
            maxiter=self.maxiter,
            with_intercept=with_intercept
        )

        self.estimator.fit(y, X=X, fh=fh)
        return self

    def predict(self, fh: ForecastingHorizon = None, X: pd.DataFrame = None):
        prediction = super().predict(fh, X)

        if self.d > 0 and fh is not None and self.estimator.fh[0] == fh.to_pandas()[0]:
            # ARIMA uses the last self.d terms for differencing. In case that the training
            # data are predicted again, the first self.d terms are missing and set to nan
            prediction = prediction.backfill()

        return prediction

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: False,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        # order
        p = UniformIntegerHyperparameter('p', lower=0, upper=5, default_value=1)
        d = UniformIntegerHyperparameter('d', lower=0, upper=2, default_value=0)
        q = UniformIntegerHyperparameter('q', lower=0, upper=5, default_value=0)

        # seasonal_order
        P = UniformIntegerHyperparameter('P', lower=0, upper=2, default_value=0)
        D = UniformIntegerHyperparameter('D', lower=0, upper=1, default_value=0)
        Q = UniformIntegerHyperparameter('Q', lower=0, upper=2, default_value=0)
        sp = CategoricalHyperparameter('sp', choices=[0, 2, 4, 7, 12], default_value=0)

        P_depends_on_sp = InCondition(P, sp, [2, 4, 7, 12])
        D_depends_on_sp = InCondition(D, sp, [2, 4, 7, 12])
        Q_depends_on_sp = InCondition(Q, sp, [2, 4, 7, 12])

        p_must_be_smaller_than_sp = ForbiddenGreaterThanRelation(sp, p)
        q_must_be_smaller_than_sp = ForbiddenGreaterThanRelation(sp, q)

        maxiter = Constant('maxiter', 50)
        with_intercept = Constant('with_intercept', 'True')

        cs = ConfigurationSpace()
        cs.add_hyperparameters([p, d, q, P, D, Q, sp, maxiter, with_intercept])
        cs.add_conditions([P_depends_on_sp, D_depends_on_sp, Q_depends_on_sp])
        cs.add_forbidden_clauses([p_must_be_smaller_than_sp, q_must_be_smaller_than_sp])

        return cs
