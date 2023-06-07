from typing import Union

import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeTransformer
from autosktime.pipeline.util import Int64Index, frequency_to_sp


class DeseasonalizerComponent(AutoSktimeTransformer):
    from sktime.transformations.series.detrend import Deseasonalizer

    _estimator_class = Deseasonalizer

    def __init__(self, model: str = 'additive', sp: int = 1, random_state: np.random.RandomState = None):
        super().__init__()
        self.model = model
        self.sp = sp
        self.random_state = random_state

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        self.estimator = self._estimator_class(model=self.model, sp=self.sp)
        self.estimator.fit(X=X, y=y)
        return self

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
        model = CategoricalHyperparameter('model', ['additive', 'multiplicative'], default_value='additive')
        sp = CategoricalHyperparameter('sp', frequency_to_sp(dataset_properties.frequency))

        cs = ConfigurationSpace()
        cs.add_hyperparameters([model, sp])
        return cs
