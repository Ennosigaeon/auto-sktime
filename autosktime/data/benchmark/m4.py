import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from six.moves import urllib

INPUT_MAPPINGS = {
    'H': {'file': 'Hourly', 'output_size': 48, 'seasonality': 24},
    'D': {'file': 'Daily', 'output_size': 14, 'seasonality': 7},
    'W': {'file': 'Weekly', 'output_size': 13, 'seasonality': 52},
    'M': {'file': 'Monthly', 'output_size': 18, 'seasonality': 12},
    'Q': {'file': 'Quarterly', 'output_size': 8, 'seasonality': 4},
    'Y': {'file': 'Yearly', 'output_size': 6, 'seasonality': 1},
    'A': {'file': 'Yearly', 'output_size': 6, 'seasonality': 1}
}


def _download_and_cache(filename: str, cache_dir: str):
    data_directory = os.path.join(cache_dir, 'm4')
    train_directory = os.path.join(data_directory, 'Train')
    test_directory = os.path.join(data_directory, 'Test')

    if not os.path.exists(data_directory):
        os.mkdir(data_directory)
    if not os.path.exists(train_directory):
        os.mkdir(train_directory)
    if not os.path.exists(test_directory):
        os.mkdir(test_directory)

    filepath = os.path.join(data_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(
            f'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/{filename}', filepath
        )
    return filepath


def load_timeseries(dataset_name: str, cache_dir: str = f'{Path.home()}/.cache/') -> Tuple[pd.Series, pd.Series]:
    _download_and_cache(f'Train/{INPUT_MAPPINGS[dataset_name[0]]["file"]}-train.csv', cache_dir)
    _download_and_cache(f'Test/{INPUT_MAPPINGS[dataset_name[0]]["file"]}-test.csv', cache_dir)
    m4info_filename = _download_and_cache('M4-info.csv', cache_dir)
    info = pd.read_csv(m4info_filename)

    data_directory = os.path.join(cache_dir, 'm4')
    train_directory = os.path.join(data_directory, 'Train')
    test_directory = os.path.join(data_directory, 'Test')

    # freq = info[info['M4id'] == dataset_name]['Frequency'][0]
    sp = info[info['M4id'] == dataset_name]['SP'].iloc[0][0]
    start_date = info[info['M4id'] == dataset_name]['StartingDate'].iloc[0]

    train_df = pd.read_csv(os.path.join(train_directory, f'{INPUT_MAPPINGS[dataset_name[0]]["file"]}-train.csv'))
    train_df = train_df[train_df['V1'] == dataset_name].iloc[:, 1:].T
    train_df = train_df.dropna()
    train_df = train_df.squeeze()
    train_df.index = pd.period_range(start=start_date, periods=len(train_df), freq=sp)

    test_df = pd.read_csv(os.path.join(test_directory, f'{INPUT_MAPPINGS[dataset_name[0]]["file"]}-test.csv'))
    test_df = test_df[test_df['V1'] == dataset_name].iloc[:, 1:].T
    test_df = test_df.dropna()
    test_df = test_df.squeeze()
    test_df.index = pd.period_range(start=train_df.index[-1] + 1, periods=len(test_df), freq=sp)

    return train_df, test_df


def _seasonality_test(past_ts_data: np.array, season_length: int) -> bool:
    """
    Test the time-series for seasonal patterns by performing a 90% auto-correlation test:

    As described here: https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
    Code based on: https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R
    """
    critical_z_score = 1.645  # corresponds to 90% confidence interval
    if len(past_ts_data) < 3 * season_length:
        return False
    else:
        # calculate auto-correlation for lags up to season_length
        auto_correlations = sm.tsa.stattools.acf(past_ts_data, fft=False, nlags=season_length)
        auto_correlations[1:] = 2 * auto_correlations[1:] ** 2
        limit = (
                critical_z_score
                / np.sqrt(len(past_ts_data))
                * np.sqrt(np.cumsum(auto_correlations))
        )
        is_seasonal = abs(auto_correlations[season_length]) > limit[season_length]

    return is_seasonal


def naive_2(
        y_train: pd.Series,
) -> pd.Series:
    """
    Make seasonality adjusted time series prediction.

    As described here: https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
    Code based on: https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R
    """
    sp = y_train.index.freqstr.split('-')[0]
    prediction_length = INPUT_MAPPINGS[sp]['output_size']
    season_length = INPUT_MAPPINGS[sp]['seasonality']
    has_seasonality = season_length > 1 and _seasonality_test(y_train, season_length)

    # it has seasonality, then calculate the multiplicative seasonal component
    if has_seasonality:
        seasonal_decomposition = sm.tsa.seasonal_decompose(x=y_train, period=season_length, model="multiplicative"
                                                           ).seasonal
        seasonality_normed_context = y_train / seasonal_decomposition

        last_period = seasonal_decomposition[-season_length:]
        num_required_periods = (prediction_length // len(last_period)) + 1

        multiplicative_seasonal_component = np.tile(last_period, num_required_periods)[:prediction_length]
    else:
        seasonality_normed_context = y_train
        multiplicative_seasonal_component = np.ones(prediction_length)  # i.e. no seasonality component

    # calculate naive forecast: (last value prediction_length times)
    naive_forecast = np.ones(prediction_length) * seasonality_normed_context.values[-1]

    index = pd.period_range(start=y_train.index[-1], periods=len(multiplicative_seasonal_component), freq=sp)
    forecast = pd.Series(np.mean(naive_forecast) * multiplicative_seasonal_component, index=index)

    return forecast


