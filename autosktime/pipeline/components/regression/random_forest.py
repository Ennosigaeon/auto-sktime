from typing import Union

import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeRegressionAlgorithm, COMPONENT_PROPERTIES
from autosktime.pipeline.util import Int64Index
from autosktime.util.backend import ConfigId
from autosktime.util.common import check_none, check_for_bool


class RandomForestComponent(AutoSktimeRegressionAlgorithm):

    def __init__(
            self,
            criterion: str = 'squared_error',
            max_features: Union[float, int] = 1.,
            max_depth: int = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            min_weight_fraction_leaf: float = 0.,
            bootstrap: bool = True,
            max_leaf_nodes: int = None,
            min_impurity_decrease: float = 0.,
            random_state: np.random.RandomState = None,
            n_jobs: int = 1,
            iterations: int = None,
            config_id: ConfigId = None
    ):
        super().__init__()
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.iterations = iterations
        self.config_id = config_id

    def _set_model(self, iterations: int):
        from sklearn.ensemble import RandomForestRegressor

        self.max_depth = None if check_none(self.max_depth) else int(self.max_depth)
        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.max_features = float(self.max_features)
        self.bootstrap = check_for_bool(self.bootstrap)
        self.max_leaf_nodes = None if check_none(self.max_leaf_nodes) else int(self.max_leaf_nodes)
        self.min_impurity_decrease = float(self.min_impurity_decrease)

        self.estimator = RandomForestRegressor(
            n_estimators=iterations,
            criterion=self.criterion,
            max_features=self.max_features,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            bootstrap=self.bootstrap,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            warm_start=True
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        iterations = self.get_iterations()
        self._set_model(iterations)
        return self._fit(X, y)

    def _update(self):
        self.estimator.n_estimators = self.get_iterations()

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()

        # noinspection PyUnresolvedReferences
        self.estimator.fit(X, y)
        return self

    def get_max_iter(self):
        return 128

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
        criterion = CategoricalHyperparameter('criterion', ['squared_error', 'absolute_error', 'poisson'])

        max_features = UniformFloatHyperparameter('max_features', 0.1, 1.0, default_value=1.0)

        max_depth = UnParametrizedHyperparameter('max_depth', 15)
        # min_samples_split = UniformIntegerHyperparameter('min_samples_split', 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter('min_samples_leaf', 1, 20, default_value=1)
        min_weight_fraction_leaf = UnParametrizedHyperparameter('min_weight_fraction_leaf', 0.)
        max_leaf_nodes = UnParametrizedHyperparameter('max_leaf_nodes', 'None')
        min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)
        # bootstrap = CategoricalHyperparameter('bootstrap', ['True', 'False'])

        cs.add_hyperparameters([criterion, max_features, max_depth, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease])

        return cs
