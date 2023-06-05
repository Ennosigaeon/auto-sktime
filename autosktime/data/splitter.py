from abc import ABC
from itertools import chain
from typing import Optional, Dict, Type, Union, List, Tuple

import numpy as np
import pandas as pd
from pandas import MultiIndex
from sklearn.model_selection import train_test_split, KFold
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import (
    SingleWindowSplitter as SingleWindowSplitter_,
    SlidingWindowSplitter as SlidingWindowSplitter_
)
# noinspection PyProtectedMember
from sktime.forecasting.model_selection._split import ACCEPTED_Y_TYPES, SPLIT_GENERATOR_TYPE, \
    BaseSplitter as BaseSplitter_

BaseSplitter = BaseSplitter_


class TemporalHoldoutSplitter(SingleWindowSplitter_):

    def __init__(self, fh: float = 0.2):
        super().__init__(1)
        self.fh_ = fh
        self._is_init = False

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        n_timepoints = y.shape[0]
        test_size = np.ceil(n_timepoints * self.fh_)
        self.fh = ForecastingHorizon(np.arange(1, test_size, dtype=int))

        return super()._split(y)


class SlidingWindowSplitter(SlidingWindowSplitter_):

    def __init__(self, folds: int = 4, fh: float = 0.2):
        super().__init__()
        self.fh_ = fh
        self.folds = folds
        self._is_init = False

    def _init(self, y: ACCEPTED_Y_TYPES):
        if not self._is_init:
            n_timepoints = y.shape[0] / self.folds

            train_size = int(n_timepoints * (1 - self.fh_))
            test_size = int(n_timepoints * self.fh_)

            self.window_length = train_size
            self.fh_ = np.arange(1, test_size + 1, dtype=int)
            self.step_length = train_size + test_size

            self._is_init = True

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        self._init(y)
        return super()._split(y)

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        self._init(y)
        return super().get_cutoffs(y)


class PanelSplitter(BaseSplitter, ABC):

    def split(self, y: ACCEPTED_Y_TYPES) -> SPLIT_GENERATOR_TYPE:
        y_index = self._coerce_to_index(y)
        if isinstance(y_index, pd.MultiIndex):
            y_index = y_index.remove_unused_levels()
            groups = y_index.get_level_values(0)
            top_level = y_index.levels[0]

            splits = []
            for train, test in self._split(top_level):
                splits.append((
                    np.where(groups.isin(train))[0], np.where(groups.isin(test))[0]
                ))
            return iter(splits)
        else:
            return self._split(y_index)


class PanelHoldoutSplitter(PanelSplitter):

    def __init__(self, fh: float = 0.2, random_state: Union[np.random.RandomState, int] = None):
        super().__init__(1)
        self.fh_ = fh
        self.random_state = random_state

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        y_train, y_test = train_test_split(y, test_size=self.fh_, random_state=self.random_state)
        return iter([(y_train, y_test)])

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
        return 1

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        pass


class PanelCVSplitter(PanelSplitter):

    def __init__(self, fh: int = 4, shuffle: bool = True, random_state: Union[np.random.RandomState, int] = None):
        super().__init__(1)
        self.fh_ = fh
        self.shuffle = shuffle
        self.random_state = random_state

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        cv = KFold(self.fh_, shuffle=self.shuffle, random_state=self.random_state)
        return map(lambda t: (y[t[0]], y[t[1]]), cv.split(y))

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
        return self.fh_

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        pass


class PreSplittedPanelSplitter(PanelSplitter):

    def __init__(self, train_ids: List[pd.Series], test_ids: List[pd.Series],
                 random_state: Union[np.random.RandomState, int] = None):
        super().__init__(1)
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.random_state = random_state

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        return map(lambda t: (t[0].values, t[1].values), zip(self.train_ids, self.test_ids))

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
        return len(self.train_ids)

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        pass


class NoSplitter(SingleWindowSplitter_):

    def __init__(self, fh: float = 0):
        super().__init__(1)

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        return iter([(y.values, y.values)])


splitter_types: Dict[str, Type[BaseSplitter]] = {
    'temporal-holdout': TemporalHoldoutSplitter,
    'sliding-window': SlidingWindowSplitter,
    'panel-holdout': PanelHoldoutSplitter,
    'panel-cv': PanelCVSplitter,
    'panel-pre': PreSplittedPanelSplitter,
}


def _validate_input(*dfs) -> pd.MultiIndex:
    index = dfs[0].index
    if not np.all([df.index == index for df in dfs]):
        raise ValueError('All dataframes must share same index')

    if not isinstance(index, pd.MultiIndex):
        raise ValueError(f'Only {type(pd.MultiIndex)} is supported got {type(index)}')
    return index


def multiindex_split(
        *dfs: Union[pd.DataFrame, pd.Series],
        train: pd.Series,
        test: pd.Series,
):
    return list(
        chain.from_iterable(
            (_safe_index(a, train), _safe_index(a, test)) for a in dfs
        )
    )


def _safe_index(df: Union[pd.DataFrame, pd.Series], idx) -> Union[pd.DataFrame, pd.Series]:
    try:
        sub_df = df.loc[idx]
    except KeyError:
        sub_df = df.iloc[idx]
    index = sub_df.index.remove_unused_levels() if isinstance(df.index, MultiIndex) else sub_df.index
    sub_df.index = index
    return sub_df


def get_ensemble_data(y: pd.Series, X: pd.DataFrame, splitter: BaseSplitter) -> Tuple[pd.Series, pd.DataFrame]:
    _, ens = next(splitter.split(y))

    y_idx = _safe_index(y, ens)
    if X is None:
        X_idx = None
    else:
        X_idx = _safe_index(X, ens)

    return y_idx, X_idx
