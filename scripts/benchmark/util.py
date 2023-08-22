from typing import Optional

import numpy as np
import pandas as pd


def fix_frequency(y: pd.Series, X_train: Optional[pd.DataFrame], X_test: Optional[pd.DataFrame]):
    freq = infer_frequency(y.index)

    if isinstance(y.index, pd.MultiIndex):
        index = transform_multiindex(y.index)
        X_train_index = transform_multiindex(X_train.index) if X_train is not None else None
        X_test_index = transform_multiindex(X_test.index) if X_test is not None else None
    else:
        index = y.index.to_period(freq)
        X_train_index = X_train.index.to_period(freq) if X_train is not None else None
        X_test_index = X_test.index.to_period(freq) if X_test is not None else None

    y.index = index
    if X_train is not None:
        X_train.index = X_train_index
    if X_test is not None:
        X_test.index = X_test_index


def transform_multiindex(index: pd.MultiIndex) -> pd.MultiIndex:
    df = index.to_frame()
    series = index.get_level_values(0)
    index_ = []
    for l0 in index.levels[0]:
        index_ += pd.RangeIndex(0, stop=len(df.loc[l0])).tolist()
    return pd.MultiIndex.from_tuples(zip(series, index_))


def infer_frequency(index: pd.Index) -> str:
    if isinstance(index, pd.MultiIndex):
        index = index.to_frame().loc[index.levels[0][0]].index

    orig_freq = pd.infer_freq(index)
    # See https://github.com/pandas-dev/pandas/issues/38914
    if orig_freq == 'MS':
        freq = 'M'
    elif orig_freq == 'AS-JAN':
        freq = 'Y'
    elif orig_freq.startswith('QS'):
        freq = 'Q'
    else:
        freq = orig_freq
    return freq


def generate_fh(index: pd.Index, fh: int) -> pd.Index:
    if isinstance(index, pd.MultiIndex):
        df = index.to_frame()
        series = np.repeat(index.levels[0], fh)
        index_ = []
        for l0 in index.levels[0]:
            index_ += pd.RangeIndex(len(df.loc[l0]) + 1, stop=len(df.loc[l0]) + fh + 1).tolist()
        return pd.MultiIndex.from_tuples(zip(series, index_))
    elif isinstance(index, pd.DatetimeIndex):
        return pd.PeriodIndex(pd.date_range(index[-1], periods=fh + 1, freq=infer_frequency(index))[1:])
    else:
        return pd.PeriodIndex(pd.date_range(index.to_timestamp()[-1], periods=fh + 1, freq=index.freq)[1:])
