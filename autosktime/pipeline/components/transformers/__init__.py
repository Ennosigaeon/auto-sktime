import os
from collections import OrderedDict
from typing import Dict, Type, Union

import pandas as pd

from autosktime.constants import SUPPORTED_INDEX_TYPES, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, find_components, AutoSktimeChoice, \
    AutoSktimeTransformer
from autosktime.pipeline.components.identity import IdentityComponent

_transformers_directory = os.path.split(__file__)[0]
_transformers = find_components(__package__, _transformers_directory, AutoSktimeTransformer)


class TransformerChoice(AutoSktimeChoice, AutoSktimeTransformer):

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None):
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: False,
            IGNORES_EXOGENOUS_X: True,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }

    @classmethod
    def get_components(cls) -> Dict[str, Type[AutoSktimeComponent]]:
        components: Dict[str, Type[AutoSktimeComponent]] = OrderedDict()
        components.update(_transformers)
        components['noop'] = IdentityComponent
        return components

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        if self.estimator is None:
            raise NotImplementedError
        # noinspection PyUnresolvedReferences
        return self.estimator.fit(X, y=y)
