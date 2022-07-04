import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeRegressionAlgorithm, COMPONENT_PROPERTIES
from autosktime.pipeline.util import Int64Index


class KNearestNeighborsRegressor(AutoSktimeRegressionAlgorithm):
    def __init__(
            self,
            n_neighbors: int = 5,
            weights: str = 'uniform',
            p: int = 2,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.neighbors import KNeighborsRegressor
        self.n_neighbors = int(self.n_neighbors)
        self.p = int(self.p)

        self.estimator = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p)

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

        n_neighbors = UniformIntegerHyperparameter("n_neighbors", lower=1, upper=100, log=True, default_value=5)
        weights = CategoricalHyperparameter("weights", choices=["uniform", "distance"])
        p = CategoricalHyperparameter("p", choices=[2, 1])

        cs.add_hyperparameters([n_neighbors, weights, p])

        return cs
