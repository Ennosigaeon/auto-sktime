import os.path

import copy
import glob
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

from autosktime.data.benchmark.base import Benchmark

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class PHME20Benchmark(Benchmark):

    def __init__(
            self,
            folds: int = 10,
            base_dir: str = f'{Path(__location__)}/data/PHME20DataChallenge/',
    ):
        self.folds = folds
        self.base_dir = base_dir
        self.start = 5

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        meta_data = {
            1: ['45-53', 0.4],
            2: ['45-53', 0.4],
            3: ['45-53', 0.4],
            4: ['45-53', 0.4],
            5: ['45-53', 0.425],
            6: ['45-53', 0.425],
            7: ['45-53', 0.425],
            8: ['45-53', 0.425],
            9: ['45-53', 0.45],
            10: ['45-53', 0.45],
            11: ['45-53', 0.45],
            12: ['45-53', 0.45],
            13: ['45-53', 0.475],
            14: ['45-53', 0.475],
            15: ['45-53', 0.475],
            16: ['45-53', 0.475],
            33: ['63-75', 0.4],
            34: ['63-75', 0.4],
            35: ['63-75', 0.4],
            36: ['63-75', 0.4],
            37: ['63-75', 0.425],
            38: ['63-75', 0.425],
            39: ['63-75', 0.425],
            40: ['63-75', 0.425],
            41: ['63-75', 0.45],
            42: ['63-75', 0.45],
            43: ['63-75', 0.45],
            44: ['63-75', 0.45],
            45: ['63-75', 0.475],
            46: ['63-75', 0.475],
            47: ['63-75', 0.475],
            48: ['63-75', 0.475]
        }

        complete = []
        for file in glob.glob(os.path.join(self.base_dir, '*', '*', 'Sample*.csv')):
            id = int(file[-6:-4])
            df = _read_single_experiment(file, *meta_data[id])
            df.index = pd.MultiIndex.from_tuples([(id, k) for k in df.index], names=['experiment', 'timestamp'])
            complete.append(df)

        X = pd.concat(complete).sort_index()
        y = X[['RUL']]
        X.drop(columns=['RUL'], inplace=True)

        return X, y

    def get_train_test_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                   44, 45, 46, 47, 48]

        test_folds = []
        train_folds = []
        val_folds = []
        for i in range(10):
            train = copy.copy(samples)
            del train[i % 4::4]
            train, val = train_test_split(train, test_size=0.2, random_state=42)

            train_folds.append(train)
            val_folds.append(val)
            test_folds.append(samples[i % 4::4])

        return pd.DataFrame(train_folds), pd.DataFrame(val_folds), pd.DataFrame(test_folds)

    @staticmethod
    def name() -> str:
        return 'phme20'


def _read_single_experiment(file_name: str, particle_size: str, solid_ratio: float) -> pd.DataFrame:
    df = pd.read_csv(file_name, index_col='Time(s)')
    # TODO handling of categorical data is missing
    df['Particle Size (micron)'] = 0 if particle_size == '45-53' else 1
    df['Solid Ratio(%)'] = solid_ratio
    df['RUL'] = df.index.max() - df.index

    # sktime does not support Float64Index for now
    df.index = (df.index * 10).astype(int)

    return df
