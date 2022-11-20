import glob
import os.path
import pickle
import re
from pathlib import Path
from typing import Tuple

import pandas as pd

from autosktime.data.benchmark.base import Benchmark

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class RULBenchmark(Benchmark):

    def __init__(
            self,
            folds: int = 10,
            base_dir: str = f'{Path(__location__)}/data/rul/',
            cache_dir: str = f'{Path.home()}/.cache/auto-sktime/'
    ):
        self.folds = folds
        self.base_dir = base_dir
        self.cache_dir = cache_dir
        self.start = 125

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data_directory = os.path.join(self.cache_dir, 'rul')
        if not os.path.exists(data_directory):
            os.makedirs(data_directory, exist_ok=True)
        cache_file = os.path.join(data_directory, 'cache.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        overview = _read_overview(self.base_dir)

        complete = []
        for file in glob.glob(os.path.join(self.base_dir, 'Data_*.csv')):
            experiment = _read_single_experiment(file, overview)
            complete.append(experiment)

        X = pd.concat(complete).sort_index()
        y = X[['RUL']]
        X.drop(columns=['RUL'], inplace=True)

        with open(cache_file, 'wb') as f:
            pickle.dump((X, y), f)

        return X, y

    def get_train_test_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data_directory = os.path.join(self.base_dir, 'splits')

        train = pd.read_csv(os.path.join(data_directory, f'idx_train_{self.folds}x.csv'), header=None)
        val = pd.read_csv(os.path.join(data_directory, f'idx_val_{self.folds}x.csv'), header=None)
        test = pd.read_csv(os.path.join(data_directory, f'idx_test_{self.folds}x.csv'), header=None)

        return train, val, test

    @staticmethod
    def name() -> str:
        return 'rul'


def _read_overview(base_dir: str) -> pd.DataFrame:
    candidates = glob.glob(os.path.join(base_dir, '_Overview_*.csv'))
    assert len(candidates) == 1
    overview = pd.read_csv(candidates[0], index_col='lfd_Nummer')
    return overview


def _read_single_experiment(file_name: str, overview: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(file_name, index_col='Messpunkt')
    df = df[['Differenzdruck', 'Vorschub', 'RUL']]

    regex = r'Data_No_(\d+)_'
    match = re.search(regex, file_name)
    no = int(match.group(1))
    # TODO handling of categorical data is missing
    # const_data = overview.loc[no, ['Filter', 'Staub', 'Durchmesser']]
    const_data = overview.loc[no, ['Durchmesser']]
    const_data = pd.concat([const_data] * df.shape[0], axis=1).T
    const_data.index = df.index

    result = pd.concat([df, const_data], sort=False, axis=1)
    result.index = pd.MultiIndex.from_tuples([(no, k) for k in result.index], names=['experiment', 'timestamp'])
    return result.infer_objects()
