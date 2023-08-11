from typing import Optional

import pandas as pd


def fix_frequency(y: pd.Series, X_train: Optional[pd.DataFrame], X_test: Optional[pd.DataFrame]):
    freq = infer_frequency(y.index)
    y.index = y.index.to_period(freq)
    if X_train is not None:
        X_train.index = X_train.index.to_period(freq)
    if X_test is not None:
        X_test.index = X_test.index.to_period(freq)


def infer_frequency(index: pd.Index) -> str:
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


def generate_fh(index: pd.Index, fh: int):
    if isinstance(index, pd.DatetimeIndex):
        return pd.PeriodIndex(pd.date_range(index[-1], periods=fh + 1, freq=infer_frequency(index))[1:])
    else:
        return pd.PeriodIndex(pd.date_range(index.to_timestamp()[-1], periods=fh + 1, freq=index.freq)[1:])
