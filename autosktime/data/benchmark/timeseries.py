import glob
import os
import pathlib
from pathlib import Path
from typing import Tuple, Union, List, Optional

import pandas as pd

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

from sktime.forecasting.model_selection import temporal_train_test_split

from scripts.benchmark.util import infer_frequency


def load_timeseries(
        fh: Optional[int] = None,
        dataset_name: Union[str, List[str]] = None
) -> List[Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, str, int]]:
    if isinstance(dataset_name, str):
        files = [f'{Path(__location__)}/data/timeseries/{dataset_name}.csv']
    elif isinstance(dataset_name, list):
        files = [f'{Path(__location__)}/data/timeseries/{d}.csv' for d in dataset_name]
    else:
        files = sorted(glob.glob(f'{Path(__location__)}/data/timeseries/*.csv'), key=str.casefold)

    res = []
    for file in files:
        df = pd.read_csv(file, parse_dates=['index'])
        if 'series' in df.columns:
            df.index = pd.MultiIndex.from_frame(df[['series', 'index']])
            df = df.drop(columns=['series', 'index'])
        else:
            df.index = df['index']
            df = df.drop(columns=['index'])

        df = df.reindex(sorted(df.columns), axis=1)

        if fh is None:
            freq = infer_frequency(df.index)
            if freq == 'H':
                fh_ = 24
            elif freq == 'D':
                fh_ = 7
            elif freq.startswith('Q'):
                fh_ = 4
            elif freq.endswith('T'):
                fh_ = int(freq[:-1])
            else:
                fh_ = 12
        else:
            fh_ = fh

        if 'y' in df.columns:
            y = df[['y']]
            y_train, y_test = temporal_train_test_split(y, test_size=fh_)
            if len(df.columns) == 1:
                X_train, X_test = None, None
            else:
                df = df.drop(columns=['y'])
                X_train, X_test = temporal_train_test_split(df, test_size=fh_)
        else:
            y_train, y_test = temporal_train_test_split(df, test_size=fh_)
            X_train, X_test = None, None
        res.append((y_train, y_test, X_train, X_test, pathlib.Path(file).name, fh_))
    return res
