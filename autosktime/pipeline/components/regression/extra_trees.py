import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeRegressionAlgorithm, COMPONENT_PROPERTIES
from autosktime.pipeline.util import Int64Index
from autosktime.util.common import check_none, check_for_bool


class ExtraTreesRegressorComponent(AutoSktimeRegressionAlgorithm):
    def __init__(
            self,
            criterion: str = 'squared_error',
            min_samples_leaf: int = 1,
            min_samples_split: int = 2,
            max_features: str = 'auto',
            bootstrap: bool = False,
            max_leaf_nodes: int = None,
            max_depth: int = None,
            min_weight_fraction_leaf: float = 0.0,
            min_impurity_decrease: float = 0.0,
            n_jobs: int = 1,
            random_state: np.random.RandomState = None,
            verbose: int = 0,

            desired_iterations: int = None
    ):
        super().__init__()
        self.n_estimators = self.get_max_iter()
        self.criterion = criterion
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.desired_iterations = desired_iterations

    def get_max_iter(self):
        return 100

    def _set_model(self, iterations: int):
        from sklearn.ensemble import ExtraTreesRegressor

        self.max_depth = None if check_none(self.max_depth) else int(self.max_depth)
        self.max_leaf_nodes = None if check_none(self.max_leaf_nodes) else int(self.max_leaf_nodes)

        self.min_samples_leaf = int(self.min_samples_leaf)
        self.min_samples_split = int(self.min_samples_split)
        self.max_features = float(self.max_features)
        self.min_impurity_decrease = float(self.min_impurity_decrease)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
        self.bootstrap = check_for_bool(self.bootstrap)
        self.n_jobs = int(self.n_jobs)
        self.verbose = int(self.verbose)

        self.estimator = ExtraTreesRegressor(n_estimators=iterations,
                                             criterion=self.criterion,
                                             max_depth=self.max_depth,
                                             min_samples_split=self.min_samples_split,
                                             min_samples_leaf=self.min_samples_leaf,
                                             bootstrap=self.bootstrap,
                                             max_features=self.max_features,
                                             max_leaf_nodes=self.max_leaf_nodes,
                                             min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                             min_impurity_decrease=self.min_impurity_decrease,
                                             n_jobs=self.n_jobs,
                                             verbose=self.verbose,
                                             random_state=self.random_state,
                                             warm_start=True)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        iterations = self.desired_iterations or self.get_max_iter()
        self._set_model(iterations)
        return self._fit(X, y)

    def update(self, X: pd.DataFrame, y: pd.Series, n_iter: int = 1):
        if self.estimator is None:
            self._set_model(n_iter)
        else:
            self.estimator.n_estimators = min(n_iter, self.estimator.n_estimators)
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
        cs = ConfigurationSpace()

        criterion = CategoricalHyperparameter('criterion', ['squared_error', 'absolute_error'])
        max_features = UniformFloatHyperparameter('max_features', 0.1, 1.0, default_value=1)

        max_depth = UnParametrizedHyperparameter(name='max_depth', value='None')
        min_weight_fraction_leaf = UnParametrizedHyperparameter('min_weight_fraction_leaf', 0.)
        max_leaf_nodes = UnParametrizedHyperparameter('max_leaf_nodes', 'None')

        min_samples_split = UniformIntegerHyperparameter('min_samples_split', 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter('min_samples_leaf', 1, 20, default_value=1)
        min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)

        bootstrap = CategoricalHyperparameter('bootstrap', ['False', 'True'])

        cs.add_hyperparameters([
            criterion, max_features, max_depth, max_leaf_nodes, min_samples_split, min_samples_leaf,
            min_impurity_decrease, min_weight_fraction_leaf, bootstrap
        ])

        return cs
