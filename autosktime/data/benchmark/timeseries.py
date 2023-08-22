import glob
import os
import pathlib
from pathlib import Path
from typing import Tuple, Union, List

import pandas as pd

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

from sktime.forecasting.model_selection import temporal_train_test_split


def load_timeseries(
        fh: int = 12,
        dataset_name: Union[str, List[str]] = None
) -> List[Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, str]]:
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

        if 'y' in df.columns:
            y = df[['y']]
            y_train, y_test = temporal_train_test_split(y, test_size=fh)
            if len(df.columns) == 1:
                X_train, X_test = None, None
            else:
                df = df.drop(columns=['y'])
                X_train, X_test = temporal_train_test_split(df, test_size=fh)
        else:
            y_train, y_test = temporal_train_test_split(df, test_size=fh)
            X_train, X_test = None, None
        res.append((y_train, y_test, X_train, X_test, pathlib.Path(file).name))
    return res
