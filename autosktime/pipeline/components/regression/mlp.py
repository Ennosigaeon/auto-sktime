import copy
import warnings

import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    UnParametrizedHyperparameter, Constant, CategoricalHyperparameter
from sklearn.exceptions import ConvergenceWarning

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeRegressionAlgorithm, COMPONENT_PROPERTIES
from autosktime.pipeline.util import Int64Index
from autosktime.util.backend import ConfigId
from autosktime.util.common import check_for_bool


class MLPClassifier(AutoSktimeRegressionAlgorithm):
    def __init__(
            self,
            hidden_layer_depth: int = 1,
            num_nodes_per_layer: int = 32,
            activation: str = 'relu',
            alpha: float = 0.0001,
            learning_rate_init: float = 0.001,
            solver: str = 'adam',
            batch_size: int = 'auto',
            n_iter_no_change: int = 32,
            tol: float = 1e-4,
            shuffle: bool = True,
            beta_1: float = 0.9,
            beta_2: float = 0.999,
            epsilon: float = 1e-8,
            random_state: np.random.RandomState = None,
            verbose: bool = False,
            iterations: int = None,
            config_id: ConfigId = None
    ):
        super().__init__()
        self.hidden_layer_depth = hidden_layer_depth
        self.num_nodes_per_layer = num_nodes_per_layer
        self.max_iter = self.get_max_iter()
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.solver = solver
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.random_state = random_state
        self.verbose = verbose
        self.iterations = iterations
        self.config_id = config_id

    def get_max_iter(self):
        return 512

    def _set_model(self, iterations: int):
        from sklearn.neural_network import MLPRegressor

        # Need to fit at least two iterations, otherwise early stopping will not work because we cannot determine
        # whether the algorithm actually converged. The only way of finding this out is if the sgd spends fewer
        # iterations than max_iter. If max_iter == 1, it has to spend at least one iteration and will always spend at
        # least one iteration, so we cannot know about convergence.
        iterations = max(iterations, 2)

        self._fully_fit = False

        self.max_iter = int(self.max_iter)
        self.hidden_layer_depth = int(self.hidden_layer_depth)
        self.num_nodes_per_layer = int(self.num_nodes_per_layer)
        self.hidden_layer_sizes = tuple(self.num_nodes_per_layer for _ in range(self.hidden_layer_depth))
        self.activation = str(self.activation)
        self.alpha = float(self.alpha)
        self.learning_rate_init = float(self.learning_rate_init)
        self.tol = float(self.tol)
        self.n_iter_no_change = int(self.n_iter_no_change)

        try:
            self.batch_size = int(self.batch_size)
        except ValueError:
            self.batch_size = str(self.batch_size)

        self.shuffle = check_for_bool(self.shuffle)
        self.beta_1 = float(self.beta_1)
        self.beta_2 = float(self.beta_2)
        self.epsilon = float(self.epsilon)
        self.beta_1 = float(self.beta_1)
        self.verbose = int(self.verbose)

        iterations = int(np.ceil(iterations))

        self.estimator = MLPRegressor(
            max_iter=iterations,
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            shuffle=self.shuffle,
            random_state=copy.copy(self.random_state),
            verbose=self.verbose,
            warm_start=True,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            beta_1=self.beta_2,
            beta_2=self.beta_1,
            epsilon=self.epsilon,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        iterations = self.get_iterations()
        self._set_model(iterations)
        return self._fit(X, y)

    def _update(self):
        self.estimator.max_iter = self.get_iterations()

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)

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
        cs = ConfigurationSpace()
        hidden_layer_depth = UniformIntegerHyperparameter('hidden_layer_depth', lower=1, upper=3, default_value=1)
        num_nodes_per_layer = UniformIntegerHyperparameter('num_nodes_per_layer', lower=16, upper=264, default_value=32,
                                                           log=True)
        activation = CategoricalHyperparameter('activation', choices=['relu', 'tanh'])
        alpha = UniformFloatHyperparameter('alpha', lower=1e-7, upper=1e-1, default_value=1e-4, log=True)

        learning_rate_init = UniformFloatHyperparameter('learning_rate_init', lower=1e-4, upper=0.5, default_value=1e-3,
                                                        log=True)

        n_iter_no_change = Constant(name='n_iter_no_change', value=32)  # default=10 is too low
        tol = UnParametrizedHyperparameter(name='tol', value=1e-4)
        solver = Constant(name='solver', value='adam')

        batch_size = UnParametrizedHyperparameter(name='batch_size', value='auto')
        shuffle = UnParametrizedHyperparameter(name='shuffle', value='True')
        beta_1 = UnParametrizedHyperparameter(name='beta_1', value=0.9)
        beta_2 = UnParametrizedHyperparameter(name='beta_2', value=0.999)
        epsilon = UnParametrizedHyperparameter(name='epsilon', value=1e-8)

        cs.add_hyperparameters([
            hidden_layer_depth, num_nodes_per_layer, activation, alpha, learning_rate_init, n_iter_no_change, tol,
            solver, batch_size, shuffle, beta_1, beta_2, epsilon
        ])
        return cs
