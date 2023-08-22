from collections.abc import MutableMapping
from typing import Any, Dict, Optional, Union, Iterator, Type

import pandas as pd
from sktime.forecasting.model_selection._split import BaseSplitter

from autosktime.data.splitter import TemporalHoldoutSplitter


class DatasetProperties(MutableMapping):

    def __init__(self, task: int, index_type: pd.Index, splitter: Optional[BaseSplitter] = None, **kwargs):
        if isinstance(index_type, pd.MultiIndex):
            series_length = index_type.to_frame().groupby(level=0).size().min()
            index_type = index_type.levels[-1]
        else:
            series_length = index_type.size

        if isinstance(splitter, TemporalHoldoutSplitter):
            series_length -= len(splitter.fh)

        self._data = dict(
            kwargs,
            index_type=type(index_type),
            task=task,
            series_length=series_length,
            freq=index_type.freq if hasattr(index_type, 'freq') else None
        )

    @property
    def index_type(self) -> Type[pd.Index]:
        return self._data['index_type']

    @property
    def task(self) -> int:
        return self._data['task']

    @property
    def series_length(self) -> int:
        return self._data['series_length']

    @property
    def frequency(self) -> Optional[pd.offsets.BaseOffset]:
        return self._data['freq']

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

    def __init__(
            self,
            task: int,
            y: pd.Series,
            X: Optional[pd.DataFrame] = None,
            y_ens: Optional[pd.Series] = None,
            X_ens: Optional[pd.DataFrame] = None,
            y_test: Optional[pd.Series] = None,
            X_test: Optional[pd.DataFrame] = None,
            dataset_name: str = '',
            splitter: Optional[BaseSplitter] = None
    ):
        self._data = {
            'y_train': y,
            'X_train': X,
            'y_ens': y_ens,
            'X_ens': X_ens,
            'y_test': y_test,
            'X_test': X_test
        }
        self._info = {
            'task': task
        }
        self._name = dataset_name
        self.dataset_properties = DatasetProperties(task, y.index, splitter=splitter)

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
    def y_ens(self) -> pd.Series:
        return self._data['y_ens']

    @property
    def X_ens(self) -> Optional[pd.DataFrame]:
        return self._data['X_ens']

    @property
    def y_test(self) -> Optional[pd.Series]:
        return self._data['y_test']

    @property
    def X_test(self) -> Optional[pd.DataFrame]:
        return self._data['X_test']

    @property
    def info(self) -> Dict[str, Any]:
        return self._info
