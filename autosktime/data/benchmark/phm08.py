import logging
import shutil
import tempfile

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
        index_names = ['experiment', 'timestamp']
        setting_names = ['setting_1', 'setting_2', 'setting_3']
        sensor_names = ['s_{}'.format(i + 1) for i in range(0, 21)]
        col_names = index_names + setting_names + sensor_names

        train = pd.read_csv(os.path.join(self.base_dir, 'train.txt'), sep='\s+', header=None, names=col_names)
        train = _add_remaining_useful_life(train, threshold=125)

        final_test = pd.read_csv(os.path.join(self.base_dir, 'final_test.txt'), sep='\s+', header=None, names=col_names)
        # The true RUL data of PHM08 are not publicly available on purpose. We will just create the predictions are send
        # them to the original authors for scoring.
        final_test['RUL'] = np.nan
        final_test['experiment'] += train['experiment'].max()

        Xy = pd.concat((train, final_test))
        Xy.index = pd.MultiIndex.from_frame(Xy[['experiment', 'timestamp']])
        y = pd.DataFrame(Xy.pop('RUL'), columns=['RUL'])
        X = Xy.drop(columns=['experiment', 'timestamp'])

        return X, y

    def get_train_test_splits(self):
        train, val = train_test_split(np.arange(1, 219), test_size=0.2, random_state=42)
        test = np.arange(219, 654)

        return (
            pd.DataFrame([train] * self.folds),
            pd.DataFrame([val] * self.folds),
            pd.DataFrame([test] * self.folds),
        )

    @staticmethod
    def name() -> str:
        return 'phm08'


def aggregate_results(input_dir: str) -> pd.DataFrame:
    import glob
    folds = []
    for file in sorted(glob.glob(os.path.join(input_dir, '*', 'predictions.npy'))):
        df = pd.read_pickle(file)
        # TODO reset experiment id to 1...434

        df = df.rename(columns={'RUL': file.split('/')[-2]})
        folds.append(df)

    df = pd.concat(folds, axis=1)
    return df


if __name__ == '__main__':
    base_dir = os.path.join(Path(__file__).parents[3], 'scripts', 'results', 'phm08')
    df = aggregate_results(base_dir)
    df.to_csv(os.path.join(base_dir, 'aggregated_results.csv'))

    with tempfile.TemporaryDirectory() as work_dir:
        last_entries = df.loc[df.groupby(['experiment'])['timestamp'].idxmax()]

        for i in range(10):
            last_entries[f'fold_{i}'].to_csv(os.path.join(work_dir, f'{i}.csv'), index=False, header=False)

        shutil.make_archive(os.path.join(base_dir, 'phm08'), 'zip', work_dir)
