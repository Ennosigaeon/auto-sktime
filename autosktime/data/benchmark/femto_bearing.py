import os.path

import glob
import logging
import numpy as np
import pandas as pd
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

from autosktime.data.benchmark.base import Benchmark


class FemtoBenchmark(Benchmark):

    def __init__(
            self,
            folds: int = 10,
            cache_dir: str = f'{Path.home()}/.cache/auto-sktime/'
    ):
        self.folds = folds
        self.cache_dir = cache_dir
        self.start = 125
        self.logger = logging.getLogger('benchmark')

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        zip_file = os.path.join(self.cache_dir, 'femto.zip')
        data_dir = os.path.join(self.cache_dir, 'femto')

        if not os.path.exists(os.path.join(data_dir, 'df.pkl')):
            if not os.path.exists(data_dir):
                if not os.path.exists(zip_file):
                    os.makedirs(data_dir, exist_ok=True)
                    self.logger.info('Downloading data. This will take some time...')
                    urllib.request.urlretrieve(
                        'https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset/archive/refs/heads/master.zip',
                        zip_file)
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)

            train_dir = os.path.join(data_dir, 'ieee-phm-2012-data-challenge-dataset-master', 'Learning_set')
            test_dir = os.path.join(data_dir, 'ieee-phm-2012-data-challenge-dataset-master', 'Full_Test_Set')
            true_rul = {
                'Bearing1_3': 5730,
                'Bearing1_4': 339,
                'Bearing1_5': 1610,
                'Bearing1_6': 1460,
                'Bearing1_7': 7570,
                'Bearing2_3': 7530,
                'Bearing2_4': 1390,
                'Bearing2_5': 3090,
                'Bearing2_6': 1290,
                'Bearing2_7': 580,
                'Bearing3_3': 820,
            }

            chunks = []
            for folder in os.listdir(train_dir):
                chunks.append(_read_single_experiment(os.path.join(train_dir, folder)))
            for folder in os.listdir(test_dir):
                chunks.append(_read_single_experiment(os.path.join(test_dir, folder)))

            X = pd.concat(chunks).sort_index()
            X.to_pickle(os.path.join(data_dir, 'df.pkl'))
        else:
            X = pd.read_pickle(os.path.join(data_dir, 'df.pkl'))

        y = X[['RUL']]
        X.drop(columns=['RUL'], inplace=True)

        return X, y

    def get_train_test_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data_dir = os.path.join(self.cache_dir, 'femto', 'ieee-phm-2012-data-challenge-dataset-master')
        train_dir = os.path.join(data_dir, 'Learning_set')
        test_dir = os.path.join(data_dir, 'Test_set')

        test_folds = []
        train_folds = []
        val_folds = []
        for folder in os.listdir(train_dir):
            train_folds.append(int(folder[-3] + folder[-1]))
            val_folds.append(int(folder[-3] + folder[-1]))
        for folder in os.listdir(test_dir):
            test_folds.append(int(folder[-3] + folder[-1]))

        test_folds = [[11, 21], [12, 25], [13, 31], [14, 17], [15, 24], [16, 32], [22, 33], [23, 26], [27, 12],
                      [13, 31]]
        train_folds = [[27, 14, 32, 33, 25, 15, 17, 13, 26, 16, 24, 22],
                       [31, 27, 17, 32, 33, 26, 24, 15, 14, 21, 13, 11],
                       [32, 12, 21, 14, 16, 17, 26, 11, 33, 22, 24, 27],
                       [24, 13, 16, 15, 27, 25, 26, 21, 11, 31, 33, 32],
                       [33, 26, 27, 12, 22, 14, 31, 13, 23, 17, 32, 21],
                       [23, 27, 22, 21, 14, 24, 12, 17, 25, 13, 31, 11],
                       [27, 16, 13, 21, 26, 25, 23, 11, 17, 12, 24, 15],
                       [21, 22, 32, 11, 31, 14, 17, 33, 16, 15, 27, 25],
                       [33, 15, 13, 26, 22, 32, 21, 11, 17, 31, 16, 23],
                       [11, 12, 15, 24, 23, 32, 22, 17, 27, 21, 25, 16]]
        val_folds = [[23, 31, 12], [22, 23, 16], [15, 23, 25], [23, 22, 12], [16, 25, 11], [15, 33, 26], [14, 32, 31],
                     [24, 12, 13], [14, 24, 25], [14, 26, 33]]

        return pd.DataFrame(train_folds), pd.DataFrame(val_folds), pd.DataFrame(test_folds)

    @staticmethod
    def name() -> str:
        return 'femto_bearing'


def _read_single_experiment(folder: str, remainder: float = 0, step_size: int = 64) -> pd.DataFrame:
    acc = []
    for file in sorted(glob.glob(os.path.join(folder, 'acc_*.csv'))):
        # These files contain wrong timestamps. Just ignore them
        if file.endswith('Bearing1_1/acc_02121.csv') or file.endswith('Bearing1_1/acc_02122.csv'):
            continue

        df = pd.read_csv(file, names=['hour', 'minute', 'second', 'ms', 'acc_h', 'acc_v'])
        # Files may use ; as seperator instead of ,
        if pd.isna(df).any().any():
            df = pd.read_csv(file, names=['hour', 'minute', 'second', 'ms', 'acc_h', 'acc_v'], sep=';')

        df2 = df[['acc_h', 'acc_v']].rolling(step_size, step=step_size).mean().loc[step_size:]
        df2[['hour', 'minute', 'second', 'ms']] = df.loc[df2.index, ['hour', 'minute', 'second', 'ms']]
        acc.append(df2)
    acc = pd.concat(acc).reset_index(drop=True)

    index = pd.DataFrame(data={'experiment': int(folder[-3] + folder[-1]), 'timestamp': acc.index})
    acc.index = pd.MultiIndex.from_frame(index)

    # Calculate RUL
    acc['second'] = acc['second'].apply(np.ceil)
    acc['RUL'] = acc['hour'] * 3600 + acc['minute'] * 60 + acc['second'] + acc['ms'] / 1000000
    acc['RUL'] = acc['RUL'] - acc['RUL'].iloc[0] + remainder
    acc['RUL'] = (acc['RUL'].max() - acc['RUL']) / acc['RUL'].max() * 100
    acc = acc.drop(columns=['hour', 'minute', 'second', 'ms'])

    return acc
