import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeRegressionAlgorithm, COMPONENT_PROPERTIES


class GaussianProcessComponent(AutoSktimeRegressionAlgorithm):
    def __init__(
            self,
            alpha: float = 1e-8,
            thetaL: float = 1e-5,
            thetaU: float = 1e-5,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.alpha = alpha
        self.thetaL = thetaL
        self.thetaU = thetaU
        self.random_state = random_state

    def fit(self, X, y):
        import sklearn.gaussian_process

        self.alpha = float(self.alpha)
        self.thetaL = float(self.thetaL)
        self.thetaU = float(self.thetaU)

        n_features = X.shape[1]
        kernel = sklearn.gaussian_process.kernels.RBF(
            length_scale=[1.0] * n_features,
            length_scale_bounds=[(self.thetaL, self.thetaU)] * n_features
        )

        # Instantiate a Gaussian Process model
        self.estimator = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            optimizer='fmin_l_bfgs_b',
            alpha=self.alpha,
            copy_X_train=True,
            random_state=self.random_state,
            normalize_y=True
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
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        alpha = UniformFloatHyperparameter("alpha", lower=1e-14, upper=1.0, default_value=1e-8, log=True)
        thetaL = UniformFloatHyperparameter("thetaL", lower=1e-10, upper=1e-3, default_value=1e-6, log=True)
        thetaU = UniformFloatHyperparameter("thetaU", lower=1.0, upper=100000, default_value=100000.0, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([alpha, thetaL, thetaU])
        return cs
