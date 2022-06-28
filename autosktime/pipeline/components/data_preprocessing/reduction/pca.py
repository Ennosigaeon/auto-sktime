import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter

from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimePreprocessingAlgorithm
from autosktime.util.common import check_for_bool


class PCAComponent(AutoSktimePreprocessingAlgorithm):
    def __init__(self, keep_variance: float = 0.999, whiten: bool = False, random_state: np.random.RandomState = None):
        super().__init__()
        self.keep_variance = keep_variance
        self.whiten = whiten

        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        from sklearn.decomposition import PCA

        n_components = float(self.keep_variance)
        self.whiten = check_for_bool(self.whiten)

        self.estimator = PCA(n_components=n_components, whiten=self.whiten, copy=True, random_state=self.random_state)
        # noinspection PyUnresolvedReferences
        self.estimator.fit(X)

        # noinspection PyUnresolvedReferences
        if not np.isfinite(self.estimator.components_).all():
            raise ValueError('PCA found non-finite components.')

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
        keep_variance = UniformFloatHyperparameter('keep_variance', 0.5, 0.9999, default_value=0.9999)
        whiten = CategoricalHyperparameter('whiten', ['False', 'True'])
        cs = ConfigurationSpace()
        cs.add_hyperparameters([keep_variance, whiten])
        return cs
