import numpy as np
import pandas as pd
from tsmoothie import ExponentialSmoother

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimePreprocessingAlgorithm
from autosktime.pipeline.util import Int64Index
from autosktime.util.backend import ConfigId


class ExponentialSmoothing(AutoSktimePreprocessingAlgorithm):

    def __init__(
            self,
            window_len: int = 10,
            alpha: float = 0.3,
            random_state: np.random.RandomState = None,
            config_id: ConfigId = None
    ):
        super().__init__()
        self.window_len = window_len
        self.alpha = alpha
        self.config_id = config_id
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X.index, pd.MultiIndex):
            Xt_complete = []
            for idx in X.index.remove_unused_levels().levels[0]:
                X_ = X.loc[idx]
                Xt = self._transform(X_)
                Xt_complete.append(Xt)
            Xt_complete = np.concatenate(Xt_complete)
        else:
            Xt_complete = self._transform(X)

        Xt = pd.DataFrame(data=Xt_complete, columns=X.columns, index=X.index)
        return Xt

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        smoother = ExponentialSmoother(window_len=self.window_len, alpha=self.alpha)
        smoother.smooth(X.T)
        Xt = smoother.smooth_data.T

        # Pad array to keep same size
        repeats = np.ones(Xt.shape[0], dtype=int)
        repeats[0] += self.window_len
        Xt = np.repeat(Xt, repeats=repeats, axis=0)
        return Xt

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: True,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        window_len = UniformIntegerHyperparameter('window_len', lower=2, upper=10, default_value=5)
        alpha = UniformFloatHyperparameter('alpha', lower=0.1, upper=0.9, default_value=0.3)

        cs.add_hyperparameters([window_len, alpha])
        return cs
