import numpy as np
import pandas as pd

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties

from autosktime.pipeline.components.base import AutoSktimeRegressionAlgorithm, COMPONENT_PROPERTIES
from autosktime.pipeline.util import Int64Index
from autosktime.util.common import check_for_bool


class PassiveAggressiveComponent(AutoSktimeRegressionAlgorithm):
    def __init__(
            self,
            C: float = 1.0,
            fit_intercept: bool = True,
            tol: float = 1e-3,
            loss: str = 'epsilon_insensitive',
            average: bool = False,
            random_state: np.random.RandomState = None,

            desired_iterations: int = None
    ):
        super().__init__()
        self.C = C
        self.fit_intercept = fit_intercept
        self.average = average
        self.tol = tol
        self.loss = loss
        self.random_state = random_state

        self.desired_iterations = desired_iterations

    def get_max_iter(self):
        return 2048

    def _set_model(self, iterations: int):
        from sklearn.linear_model import PassiveAggressiveRegressor

        # Need to fit at least two iterations, otherwise early stopping will not work because we cannot determine
        # whether the algorithm actually converged. The only way of finding this out is if the sgd spends fewer
        # iterations than max_iter. If max_iter == 1, it has to spend at least one iteration and will always spend at
        # least one iteration, so we cannot know about convergence.
        n_iter = max(iterations, 2)

        self.average = check_for_bool(self.average)
        self.fit_intercept = check_for_bool(self.fit_intercept)
        self.tol = float(self.tol)
        self.C = float(self.C)

        self.estimator = PassiveAggressiveRegressor(
            C=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=n_iter,
            tol=self.tol,
            loss=self.loss,
            shuffle=True,
            random_state=self.random_state,
            warm_start=True,
            average=self.average,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        iterations = self.desired_iterations or self.get_max_iter()
        self._set_model(iterations)
        return self._fit(X, y)

    def update(self, X: pd.DataFrame, y: pd.Series, n_iter: int = 1):
        if self.estimator is None:
            self._set_model(n_iter)
        else:
            self.estimator.max_iter = min(n_iter, self.estimator.max_iter)
        return self._fit(X, y)

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()

        # noinspection PyUnresolvedReferences
        self.estimator.fit(X, y)
        return self

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
        C = UniformFloatHyperparameter('C', 1e-5, 10, 1.0, log=True)
        fit_intercept = UnParametrizedHyperparameter('fit_intercept', 'True')
        loss = CategoricalHyperparameter('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive'])

        tol = UniformFloatHyperparameter('tol', 1e-5, 1e-1, default_value=1e-4, log=True)
        # Note: Average could also be an Integer if > 1
        average = CategoricalHyperparameter('average', ['False', 'True'])

        cs = ConfigurationSpace()
        cs.add_hyperparameters([loss, fit_intercept, tol, C, average])
        return cs
