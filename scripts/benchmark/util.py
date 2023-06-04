import pandas as pd


def generate_fh(index: pd.Index, fh: int):
    orig_freq = pd.infer_freq(index)
    # See https://github.com/pandas-dev/pandas/issues/38914
    freq = 'M' if orig_freq == 'MS' else orig_freq

    fh = pd.PeriodIndex(pd.date_range(index[-1], periods=fh + 1, freq=freq)[1:])

    return fh
