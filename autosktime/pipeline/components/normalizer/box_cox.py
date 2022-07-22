from typing import Union

import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, \
    ForbiddenAndConjunction, ForbiddenGreaterThanRelation, ForbiddenEqualsClause
from autosktime.constants import IGNORES_EXOGENOUS_X, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeTransformer
from autosktime.pipeline.util import Int64Index


class BoxCoxComponent(AutoSktimeTransformer):
    from sktime.transformations.series.boxcox import BoxCoxTransformer

    _estimator_class = BoxCoxTransformer

    def __init__(
            self,
            lower_bound: float = -2,
            upper_bound: float = 2,
            method: str = 'mle',
            sp: int = None,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.method = method
        self.sp = sp
        self.random_state = random_state

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        self.estimator = self._estimator_class(
            bounds=(self.lower_bound, self.upper_bound),
            method=self.method,
            sp=self.sp
        )

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
        lower_bound = UniformFloatHyperparameter('lower_bound', lower=-5, upper=2, default_value=-2)
        upper_bound = UniformFloatHyperparameter('upper_bound', lower=-2, upper=5, default_value=2)
        lower_bound_smaller_than_upper_bound = ForbiddenGreaterThanRelation(lower_bound, upper_bound)

        method = CategoricalHyperparameter('method', choices=['pearsonr', 'mle', 'guerrero'],
                                           default_value='mle')
        sp = CategoricalHyperparameter('sp', choices=[0, 2, 4, 7, 12], default_value=0)

        guerrero_requires_sp_greater_zero = ForbiddenAndConjunction(
            ForbiddenEqualsClause(method, 'guerrero'),
            ForbiddenEqualsClause(sp, 0)
        )

        cs = ConfigurationSpace()
        cs.add_hyperparameters([lower_bound, upper_bound, method, sp])
        cs.add_forbidden_clauses([guerrero_requires_sp_greater_zero, lower_bound_smaller_than_upper_bound])

        return cs
