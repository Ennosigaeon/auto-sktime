import abc
from collections.abc import MutableMapping
from typing import Any, Dict, Optional, Union, Type, Iterator

import pandas as pd

from autosktime.constants import UNIVARIATE_ENDOGENOUS_FORECAST, UNIVARIATE_EXOGENOUS_FORECAST


class DatasetProperties(MutableMapping):

    def __init__(self, index_type: pd.Index = None, **kwargs):
        if isinstance(index_type, pd.MultiIndex):
            index_type = index_type.levels[-1]

        self._data = dict(
            kwargs,
            index_type=type(index_type),
        )

    @property
    def index_type(self):
        return self._data['index_type']

    def __setitem__(self, k: str, v: Any) -> None:
        self._data[k] = v

    def __delitem__(self, v: str) -> None:
        del self._data[v]

    def __getitem__(self, k: str) -> Any:
        return self._data[k]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __copy__(self) -> 'DatasetProperties':
        return DatasetProperties(**self._data)


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
        self.dataset_properties = DatasetProperties(y.index)

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
