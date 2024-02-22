import os

import numpy as np
import pandas as pd

from autosktime.data.benchmark.timeseries import load_timeseries
from autosktime.metrics import RootMeanSquaredError, MeanAbsolutePercentageError

for method in os.listdir('results2'):
    df = pd.read_csv(f'results2/{method}/_result.csv')
    df['sMAPE'] = np.inf
    df['RMSE'] = np.inf

    for y_train, y_test, _, _, ds_name, fh in load_timeseries():
        for i in range(5):
            try:
                y_pred = pd.read_csv(f'results2/{method}/{ds_name[:-4]}-{method}-{i}-prediction.csv',
                                     parse_dates=['index'])
                if 'series' in y_pred.columns:
                    y_pred.index = pd.MultiIndex.from_frame(y_pred[['series', 'index']])
                    y_pred = y_pred.drop(columns=['series', 'index'])
                else:
                    y_pred.index = y_pred['index']
                    y_pred = y_pred.drop(columns=['index'])

                rmse = RootMeanSquaredError()(y_pred.values, y_test.values)
                smape = MeanAbsolutePercentageError(symmetric=True)(y_pred.values, y_test.values)
                df.loc[(df['dataset'] == ds_name) & (df['seed'] == i), 'sMAPE'] = smape
                df.loc[(df['dataset'] == ds_name) & (df['seed'] == i), 'RMSE'] = rmse
            except FileNotFoundError:
                pass
            except Exception as a:
                pass

    df.to_csv(f'results2/{method}/_result.csv', index=False)
