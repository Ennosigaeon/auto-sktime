import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

from autosktime.data.benchmark.base import Benchmark

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class CMAPSSBenchmark(Benchmark):

    def __init__(
            self,
            number: int = 1,
            folds: int = 10,
            base_dir: str = f'{Path(__location__)}/data/CMAPSS/',
            cache_dir: str = f'{Path.home()}/.cache/auto-sktime/'
    ):
        self.number = number
        self.folds = folds
        self.base_dir = base_dir
        self.cache_dir = cache_dir
        self.start = 5

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        index_names = ['experiment', 'timestamp']
        setting_names = ['setting_1', 'setting_2', 'setting_3']
        sensor_names = ['s_{}'.format(i + 1) for i in range(0, 21)]
        col_names = index_names + setting_names + sensor_names

        train = pd.read_csv(os.path.join(self.base_dir, f'train_FD00{self.number}.txt'), sep='\s+', header=None,
                            names=col_names)
        test = pd.read_csv(os.path.join(self.base_dir, f'test_FD00{self.number}.txt'), sep='\s+', header=None,
                           names=col_names)
        y_test = pd.read_csv(os.path.join(self.base_dir, f'RUL_FD00{self.number}.txt'), sep='\s+', header=None)[0]
        y_test.index += 1

        train = _add_remaining_useful_life(train, threshold=self.get_piecewise_cutoff())
        test = _add_remaining_useful_life(test, threshold=self.get_piecewise_cutoff(), y=y_test)
        test['experiment'] *= -1

        Xy = pd.concat((train, test))
        Xy.index = pd.MultiIndex.from_frame(Xy[['experiment', 'timestamp']])
        y = pd.DataFrame(Xy.pop('RUL'), columns=['RUL'])
        X = Xy.drop(columns=['experiment', 'timestamp'])

        return X, y

    def get_train_test_splits(self):
        train, val = train_test_split(np.arange(1, 101), test_size=0.2, random_state=42)

        return (
            pd.DataFrame([train] * self.folds),
            pd.DataFrame([val] * self.folds),
            pd.DataFrame([np.arange(-1, -101, -1)] * self.folds)
        )

    def get_piecewise_cutoff(self) -> int:
        return 125

    @staticmethod
    def name() -> str:
        return 'cmapss'


def _add_remaining_useful_life(df: pd.DataFrame, threshold: float = None, y: pd.Series = None):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by='experiment')
    max_cycle = grouped_by_unit['timestamp'].max()
    if y is not None:
        max_cycle = max_cycle + y

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='experiment', right_index=True)

    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame['max_cycle'] - result_frame['timestamp']
    result_frame['RUL'] = remaining_useful_life
    if threshold is not None:
        result_frame['RUL'] = result_frame['RUL'].clip(upper=threshold)

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop('max_cycle', axis=1)
    return result_frame


class CMAPSS1Benchmark(CMAPSSBenchmark):

    def __init__(self, folds: int = 10, base_dir: str = f'{Path(__location__)}/data/CMAPSS/',
                 cache_dir: str = f'{Path.home()}/.cache/auto-sktime/'):
        super().__init__(number=1, folds=folds, base_dir=base_dir, cache_dir=cache_dir)

    @staticmethod
    def name() -> str:
        return 'cmapss_1'


class CMAPSS2Benchmark(CMAPSSBenchmark):

    def __init__(self, folds: int = 10, base_dir: str = f'{Path(__location__)}/data/CMAPSS/',
                 cache_dir: str = f'{Path.home()}/.cache/auto-sktime/'):
        super().__init__(number=2, folds=folds, base_dir=base_dir, cache_dir=cache_dir)

    @staticmethod
    def name() -> str:
        return 'cmapss_2'


class CMAPSS3Benchmark(CMAPSSBenchmark):

    def __init__(self, folds: int = 10, base_dir: str = f'{Path(__location__)}/data/CMAPSS/',
                 cache_dir: str = f'{Path.home()}/.cache/auto-sktime/'):
        super().__init__(number=3, folds=folds, base_dir=base_dir, cache_dir=cache_dir)

    @staticmethod
    def name() -> str:
        return 'cmapss_3'


class CMAPSS4Benchmark(CMAPSSBenchmark):

    def __init__(self, folds: int = 10, base_dir: str = f'{Path(__location__)}/data/CMAPSS/',
                 cache_dir: str = f'{Path.home()}/.cache/auto-sktime/'):
        super().__init__(number=4, folds=folds, base_dir=base_dir, cache_dir=cache_dir)

    @staticmethod
    def name() -> str:
        return 'cmapss_4'
