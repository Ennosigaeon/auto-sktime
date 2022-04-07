from typing import Optional

import numpy as np
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import (
    SingleWindowSplitter as SingleWindowSplitter_,
    SlidingWindowSplitter as SlidingWindowSplitter_
)
# noinspection PyProtectedMember
from sktime.forecasting.model_selection._split import ACCEPTED_Y_TYPES, SPLIT_GENERATOR_TYPE, \
    BaseSplitter as BaseSplitter_

BaseSplitter = BaseSplitter_


class HoldoutSplitter(SingleWindowSplitter_):

    def __init__(self, fh: float = 0.2):
        super().__init__(1)
        self.fh_ = fh
        self._is_init = False

    def _init(self, y: ACCEPTED_Y_TYPES):
        if not self._is_init:
            n_timepoints = y.shape[0]
            test_size = np.floor(n_timepoints * self.fh_) + 1
            self.fh = ForecastingHorizon(np.arange(1, test_size, dtype=int))
            self._is_init = True

    def split(self, y: ACCEPTED_Y_TYPES) -> SPLIT_GENERATOR_TYPE:
        self._init(y)
        return super().split(y)


class SlidingWindowSplitter(SlidingWindowSplitter_):

    def __init__(self, folds: int = 4, fh: float = 0.2):
        super().__init__()
        self.fh = fh
        self.folds = folds
        self._is_init = False

    def _init(self, y: ACCEPTED_Y_TYPES):
        if not self._is_init:
            n_timepoints = y.shape[0] / self.folds

            train_size = int(n_timepoints * (1 - self.fh))
            test_size = int(n_timepoints * self.fh)

            self.window_length = train_size
            self.fh = np.arange(1, test_size + 1, dtype=int)
            self.step_length = train_size + test_size

            self._is_init = True

    def split(self, y: ACCEPTED_Y_TYPES) -> SPLIT_GENERATOR_TYPE:
        self._init(y)
        return super().split(y)

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        self._init(y)
        return super().get_cutoffs(y)
