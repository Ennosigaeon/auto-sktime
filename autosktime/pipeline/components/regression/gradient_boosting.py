import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    UnParametrizedHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeRegressionAlgorithm, COMPONENT_PROPERTIES
from autosktime.util.common import check_none


class GradientBoostingComponent(AutoSktimeRegressionAlgorithm):

    def __init__(
            self,
            loss: str = 'squared_error',
            learning_rate: float = 0.1,
            min_samples_leaf: int = 20,
            max_depth: int = None,
            max_leaf_nodes: int = 31,
            max_bins: int = 255,
            l2_regularization: float = 0.0,
            tol: float = 1e-7,
            scoring: str = 'loss',
            n_iter_no_change=10,
            validation_fraction: float = None,
            random_state: np.random.RandomState = None,
            verbose: int = 0
    ):
        super().__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.tol = tol
        self.scoring = scoring
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.learning_rate = float(self.learning_rate)
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.max_depth = None if check_none(self.max_depth) else int(self.max_depth)
        self.max_leaf_nodes = None if check_none(self.max_leaf_nodes) else int(self.max_leaf_nodes)
        self.max_bins = int(self.max_bins)
        self.l2_regularization = float(self.l2_regularization)
        self.tol = float(self.tol)
        self.scoring = None if check_none(self.scoring) else self.scoring
        self.verbose = int(self.verbose)

        self.estimator = HistGradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            max_bins=self.max_bins,
            l2_regularization=self.l2_regularization,
            tol=self.tol,
            scoring=self.scoring,
            n_iter_no_change=self.n_iter_no_change,
            verbose=self.verbose,
            warm_start=True,
            random_state=self.random_state,
        )

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
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, pd.core.indexes.numeric.Int64Index]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        loss = UnParametrizedHyperparameter('loss', value='squared_error')
        learning_rate = UniformFloatHyperparameter('learning_rate', lower=0.01, upper=1, default_value=0.1, log=True)
        min_samples_leaf = UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=200, default_value=20,
                                                        log=True)
        max_depth = UnParametrizedHyperparameter('max_depth', value='None')
        max_leaf_nodes = UniformIntegerHyperparameter('max_leaf_nodes', lower=3, upper=2047, default_value=31, log=True)
        max_bins = Constant('max_bins', 255)
        l2_regularization = UniformFloatHyperparameter('l2_regularization', lower=1E-10, upper=1, default_value=1E-10,
                                                       log=True)

        tol = UnParametrizedHyperparameter('tol', value=1e-7)
        scoring = UnParametrizedHyperparameter('scoring', value='loss')
        n_iter_no_change = UniformIntegerHyperparameter(name='n_iter_no_change', lower=1, upper=20, default_value=10)
        validation_fraction = UniformFloatHyperparameter('validation_fraction', lower=0.01, upper=0.4,
                                                         default_value=0.1)

        cs.add_hyperparameters(
            [loss, learning_rate, min_samples_leaf, max_depth, max_leaf_nodes, max_bins, l2_regularization, tol,
             scoring, n_iter_no_change, validation_fraction])

        return cs
