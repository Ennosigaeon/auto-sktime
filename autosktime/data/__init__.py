import abc
from typing import Any, Dict

import pandas as pd


class AbstractDataManager:
    __metaclass__ = abc.ABCMeta

    def __init__(self, name: str):
        self._data = dict()  # type: Dict
        self._info = dict()  # type: Dict
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> Dict[str, pd.Series]:
        return self._data

    @property
    def info(self) -> Dict[str, Any]:
        return self._info


class TimeSeriesDataManager(AbstractDataManager):

    def __init__(self,
                 y: pd.Series,
                 task: int,
                 dataset_name: str):
        super().__init__(dataset_name)
        self.data['y_train'] = y
        self.info['task'] = task
