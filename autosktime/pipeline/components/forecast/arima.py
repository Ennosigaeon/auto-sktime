from typing import Optional, Union

from ConfigSpace.forbidden import ForbiddenLambda
from sktime.forecasting.base import ForecastingHorizon

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, \
    Constant, InCondition
from autosktime.pipeline.components.base import AutoSktimePredictor, DATASET_PROPERTIES, COMPONENT_PROPERTIES


class ARIMAComponent(AutoSktimePredictor):

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
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.sp = sp
        self.maxiter = maxiter
        self.with_intercept = with_intercept

    def fit(self, y, X=None, fh: Optional[ForecastingHorizon] = None):
        from sktime.forecasting.arima import ARIMA

        if self.d >= y.shape[0]:
            raise ValueError('Trainings data is too short ({}) for selected d ({}). Try to increase'
                             'the trainings data or decrease d'.format(y.shape[0], self.d))

        try:
            with_intercept = bool(self.with_intercept)
        except ValueError:
            with_intercept = self.with_intercept

        self.estimator = ARIMA(
            order=(self.p, self.d, self.q),
            seasonal_order=(self.P, self.D, self.Q, self.sp),
            maxiter=self.maxiter,
            with_intercept=with_intercept
        )

        self.estimator.fit(y, X=X, fh=fh)
        return self

    def predict(
            self,
            fh: Optional[ForecastingHorizon] = None,
            X=None
    ):
        prediction = super().predict(fh, X)

        if self.d > 0 and fh is not None and self.estimator.fh[0] == fh.to_pandas()[0]:
            # ARIMA uses the last self.d terms for differencing. In case that the training
            # data are predicted again, the first self.d terms are missing and set to nan
            prediction = prediction.backfill()

        return prediction

    @staticmethod
    def get_properties(dataset_properties: DATASET_PROPERTIES = None) -> COMPONENT_PROPERTIES:
        from sktime.forecasting.arima import ARIMA
        return ARIMA.get_class_tags()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DATASET_PROPERTIES = None) -> ConfigurationSpace:
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

        def _invalid_sp(hp, sp) -> bool:
            return 1 < sp <= hp

        p_must_be_smaller_than_sp = ForbiddenLambda(p, sp, _invalid_sp)
        q_must_be_smaller_than_sp = ForbiddenLambda(q, sp, _invalid_sp)

        maxiter = Constant('maxiter', 50)
        with_intercept = Constant('with_intercept', 'True')

        cs = ConfigurationSpace()
        cs.add_hyperparameters([p, d, q, P, D, Q, sp, maxiter, with_intercept])
        cs.add_conditions([P_depends_on_sp, D_depends_on_sp, Q_depends_on_sp])
        cs.add_forbidden_clauses([p_must_be_smaller_than_sp, q_must_be_smaller_than_sp])

        return cs
