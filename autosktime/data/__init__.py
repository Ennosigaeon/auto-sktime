import abc
from typing import Any, Dict, Optional, Union, Type

import pandas as pd

from autosktime.constants import UNIVARIATE_ENDOGENOUS_FORECAST, UNIVARIATE_EXOGENOUS_FORECAST


class DatasetProperties:

    def __init__(self, index_type: Type[pd.Index]):
        self.index_type = index_type


class AbstractDataManager:
    __metaclass__ = abc.ABCMeta

    def __init__(self, y: pd.Series, X: Optional[pd.DataFrame], task: int, dataset_name: str):
        self._data = {
            'y_train': y,
            'X_train': X
        }
        self._info = {
            'task': task
        }
        self._name = dataset_name
        self.dataset_properties = DatasetProperties(type(y.index))

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        return self._data

    @property
    def y(self) -> pd.Series:
        return self._data['y_train']

    @property
    def X(self) -> Optional[pd.DataFrame]:
        return self._data['X_train']

    @property
    def info(self) -> Dict[str, Any]:
        return self._info


class UnivariateTimeSeriesDataManager(AbstractDataManager):

    def __init__(self, y: pd.Series, dataset_name: str):
        super().__init__(y, None, UNIVARIATE_ENDOGENOUS_FORECAST, dataset_name)


class UnivariateExogenousTimeSeriesDataManager(AbstractDataManager):

    def __init__(self, y: pd.Series, X: pd.DataFrame, dataset_name: str):
        super().__init__(y, X, UNIVARIATE_EXOGENOUS_FORECAST, dataset_name)
