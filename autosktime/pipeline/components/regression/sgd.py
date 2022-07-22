import numpy as np
import pandas as pd
from ConfigSpace.conditions import InCondition, EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

from ConfigSpace import Constant
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeRegressionAlgorithm
from autosktime.pipeline.util import Int64Index
from autosktime.util.common import check_for_bool


class SGDComponent(AutoSktimeRegressionAlgorithm):

    def __init__(
            self,
            loss: str = 'squared_error',
            penalty: str = 'l2',
            alpha: float = 0.0001,
            fit_intercept: bool = True,
            tol: float = 1e-3,
            learning_rate: str = 'invscaling',
            l1_ratio: float = 0.15,
            epsilon: float = 0.1,
            eta0: float = 0.01,
            power_t: float = 0.5,
            average: bool = False,
            random_state: np.random.RandomState = None,

            desired_iterations: int = None
    ):
        super().__init__()
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.epsilon = epsilon
        self.eta0 = eta0
        self.power_t = power_t
        self.random_state = random_state
        self.average = average

        self.desired_iterations = desired_iterations
        self.scaler = None

    def _set_model(self, iterations: int):
        from sklearn.linear_model import SGDRegressor
        from sklearn.preprocessing import StandardScaler

        self.alpha = float(self.alpha)
        self.l1_ratio = float(self.l1_ratio) if self.l1_ratio is not None else 0.15
        self.epsilon = float(self.epsilon) if self.epsilon is not None else 0.1
        self.eta0 = float(self.eta0)
        self.power_t = float(self.power_t) if self.power_t is not None else 0.5
        self.average = check_for_bool(self.average)
        self.fit_intercept = check_for_bool(self.fit_intercept)
        self.tol = float(self.tol)

        self.estimator = SGDRegressor(max_iter=iterations,
                                      loss=self.loss,
                                      penalty=self.penalty,
                                      alpha=self.alpha,
                                      fit_intercept=self.fit_intercept,
                                      tol=self.tol,
                                      learning_rate=self.learning_rate,
                                      l1_ratio=self.l1_ratio,
                                      epsilon=self.epsilon,
                                      eta0=self.eta0,
                                      power_t=self.power_t,
                                      shuffle=True,
                                      average=self.average,
                                      random_state=self.random_state,
                                      warm_start=True)

        self.scaler = StandardScaler(copy=True)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        iterations = self.desired_iterations or self.get_max_iter()
        self._set_model(iterations)

        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.scaler.fit(y)
        return self._fit(X, y)

    def update(self, X: pd.DataFrame, y: pd.Series, n_iter: int = 1):
        if self.estimator is None:
            self._set_model(n_iter)
        else:
            self.estimator.max_iter = min(n_iter, self.estimator.max_iter)
        return self._fit(X, y)

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        y_scaled = self.scaler.transform(y)

        # Flatten: [[0], [0], [0]] -> [0, 0, 0]
        if y_scaled.ndim == 2 and y_scaled.shape[1] == 1:
            y_scaled = y_scaled.flatten()

        # noinspection PyUnresolvedReferences
        self.estimator.fit(X, y_scaled)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.estimator is None:
            raise NotImplementedError()
        # noinspection PyUnresolvedReferences
        y_pred = self.estimator.predict(X)
        tmp = self.scaler.inverse_transform(np.atleast_2d(y_pred))[0]
        return tmp

    def get_max_iter(self):
        return 1024

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        loss = CategoricalHyperparameter('loss', ['squared_error', 'huber', 'squared_epsilon_insensitive',
                                                  'epsilon_insensitive'])
        penalty = CategoricalHyperparameter('penalty', ['l2', 'l1', 'elasticnet'])
        alpha = UniformFloatHyperparameter('alpha', 1e-7, 1e-1, log=True, default_value=0.0001)
        l1_ratio = UniformFloatHyperparameter('l1_ratio', 1e-9, 1., log=True, default_value=0.15)
        fit_intercept = Constant('fit_intercept', 'True')
        tol = UniformFloatHyperparameter('tol', 1e-5, 1e-1, default_value=1e-4, log=True)
        epsilon = UniformFloatHyperparameter('epsilon', 1e-5, 1e-1, default_value=0.1, log=True)
        learning_rate = CategoricalHyperparameter('learning_rate', ['invscaling', 'optimal', 'constant'])
        eta0 = UniformFloatHyperparameter('eta0', 1e-7, 1e-1, default_value=0.01, log=True)
        power_t = UniformFloatHyperparameter('power_t', 1e-5, 1, default_value=0.25)
        average = CategoricalHyperparameter('average', ['False', 'True'])

        cs.add_hyperparameters([loss, penalty, alpha, l1_ratio, fit_intercept, tol, epsilon, learning_rate, eta0,
                                power_t, average])

        elasticnet = EqualsCondition(l1_ratio, penalty, 'elasticnet')
        epsilon_condition = InCondition(epsilon, loss, ['huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'])

        # eta0 is only relevant if learning_rate!='optimal' according to code
        # https://github.com/scikit-learn/scikit-learn/blob/0.19.X/sklearn/linear_model/sgd_fast.pyx#L603
        eta0_in_inv_con = InCondition(eta0, learning_rate, ['invscaling', 'constant'])
        power_t_condition = EqualsCondition(power_t, learning_rate, 'invscaling')

        cs.add_conditions([elasticnet, epsilon_condition, power_t_condition, eta0_in_inv_con])

        return cs
