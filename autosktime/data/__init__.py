import abc
from typing import Any, Dict, Optional, Union

import pandas as pd

from autosktime.constants import UNIVARIATE_ENDOGENOUS_FORECAST, UNIVARIATE_EXOGENOUS_FORECAST


class AbstractDataManager:
    __metaclass__ = abc.ABCMeta

    def __init__(self, name: str):
        self._data = dict()
        self._info = dict()
        self._name = name

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


class TimeSeriesDataManager(AbstractDataManager):

    def __init__(self, y: pd.Series, X: Optional[pd.DataFrame], task: int, dataset_name: str):
        super().__init__(dataset_name)
        self.data['y_train'] = y
        self.data['X_train'] = X
        self.info['task'] = task


class UnivariateTimeSeriesDataManager(TimeSeriesDataManager):

    def __init__(self, y: pd.Series, dataset_name: str):
        super().__init__(y, None, UNIVARIATE_ENDOGENOUS_FORECAST, dataset_name)


class UnivariateExogenousTimeSeriesDataManager(TimeSeriesDataManager):

    def __init__(self, y: pd.Series, X: pd.DataFrame, dataset_name: str):
        super().__init__(y, X, UNIVARIATE_EXOGENOUS_FORECAST, dataset_name)
