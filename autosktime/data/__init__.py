from collections.abc import MutableMapping
from typing import Any, Dict, Optional, Union, Iterator, Type

import pandas as pd


class DatasetProperties(MutableMapping):

    def __init__(self, task: int, index_type: pd.Index, **kwargs):
        if isinstance(index_type, pd.MultiIndex):
            index_type = index_type.levels[-1]

        self._data = dict(
            kwargs,
            index_type=type(index_type),
            task=task
        )

    @property
    def index_type(self) -> Type[pd.Index]:
        return self._data['index_type']

    @property
    def task(self) -> int:
        return self._data['task']

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


class DataManager:

    def __init__(self, task: int, y: pd.Series, X: Optional[pd.DataFrame] = None, dataset_name: str = ''):
        self._data = {
            'y_train': y,
            'X_train': X
        }
        self._info = {
            'task': task
        }
        self._name = dataset_name
        self.dataset_properties = DatasetProperties(task, y.index)

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
