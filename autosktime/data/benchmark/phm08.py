import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

from autosktime.data.benchmark.base import Benchmark
from autosktime.data.benchmark.cmapss import _add_remaining_useful_life

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class PHM08Benchmark(Benchmark):

    def __init__(
            self,
            folds: int = 10,
            base_dir: str = f'{Path(__location__)}/data/Phm08DataChallenge/',
            cache_dir: str = f'{Path.home()}/.cache/auto-sktime/'
    ):
        self.folds = folds
        self.base_dir = base_dir
        self.cache_dir = cache_dir
        self.start = 5
        self.logger = logging.getLogger('benchmark')

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # TODO
        self.logger.warning(f'Test data does not contain ground truth. Currently only using train data!')
        index_names = ['experiment', 'timestamp']
        setting_names = ['setting_1', 'setting_2', 'setting_3']
        sensor_names = ['s_{}'.format(i + 1) for i in range(0, 21)]
        col_names = index_names + setting_names + sensor_names

        train = pd.read_csv(os.path.join(self.base_dir, 'train.txt'), sep='\s+', header=None, names=col_names)
        test = pd.read_csv(os.path.join(self.base_dir, 'test.txt'), sep='\s+', header=None, names=col_names)
        final_test = pd.read_csv(os.path.join(self.base_dir, 'final_test.txt'), sep='\s+', header=None, names=col_names)
        y_test = pd.read_csv(os.path.join(self.base_dir, f'RUL_test.txt'), sep='\s+', header=None)[0]
        y_test.index += 1
        y_final_test = pd.read_csv(os.path.join(self.base_dir, f'RUL_final_test.txt'), sep='\s+', header=None)[0]
        y_final_test.index += 1

        train = _add_remaining_useful_life(train, threshold=125)
        test = _add_remaining_useful_life(test, threshold=125)
        final_test = _add_remaining_useful_life(final_test, threshold=125)
        test['experiment'] += train['experiment'].max()
        final_test['experiment'] *= -1

        Xy = pd.concat((train, test, final_test))
        Xy.index = pd.MultiIndex.from_frame(Xy[['experiment', 'timestamp']])
        y = pd.DataFrame(Xy.pop('RUL'), columns=['RUL'])
        X = Xy.drop(columns=['experiment', 'timestamp'])

        return X, y

    def get_train_test_splits(self):
        train, val = train_test_split(np.arange(1, 219), test_size=0.2, random_state=42)

        return (
            pd.DataFrame([train] * self.folds),
            pd.DataFrame([train] * self.folds),
            pd.DataFrame([val] * self.folds),
            # pd.DataFrame([np.arange(1, 219, 1)] * self.folds),
            # pd.DataFrame([np.arange(219, 437, 1)] * self.folds),
            # pd.DataFrame([np.arange(-1, -436, -1)] * self.folds)
        )

    @staticmethod
    def name() -> str:
        return 'phm08'
