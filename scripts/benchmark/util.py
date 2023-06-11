import pandas as pd


def fix_frequency(y: pd.Series):
    orig_freq = pd.infer_freq(y.index)
    # See https://github.com/pandas-dev/pandas/issues/38914
    if orig_freq == 'MS':
        freq = 'M'
    elif orig_freq.startswith('QS'):
        freq = 'Q'
    else:
        freq = orig_freq
    y.index = y.index.to_period(freq)


def generate_fh(index: pd.Index, fh: int):
    return pd.PeriodIndex(pd.date_range(index.to_timestamp()[-1], periods=fh + 1, freq=index.freq)[1:])
