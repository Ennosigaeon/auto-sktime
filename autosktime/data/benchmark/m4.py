import os
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Union, List

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
        os.makedirs(data_directory, exist_ok=True)
    if not os.path.exists(train_directory):
        os.makedirs(train_directory, exist_ok=True)
    if not os.path.exists(test_directory):
        os.makedirs(test_directory, exist_ok=True)

    filepath = os.path.join(data_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(
            f'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/{filename}', filepath
        )
    return filepath


def load_timeseries(
        dataset_name: Union[str, List[str]],
        cache_dir: str = f'{Path.home()}/.cache/auto-sktime/'
) -> Union[
    Tuple[List[pd.Series], List[pd.Series]],
    Tuple[pd.Series, pd.Series]
]:
    if isinstance(dataset_name, str):
        files = {dataset_name[0]: [dataset_name]}
    else:
        files = defaultdict(list)
        for name in dataset_name:
            files[name[0]].append(name)

    m4info_filename = _download_and_cache('M4-info.csv', cache_dir)
    info = pd.read_csv(m4info_filename)

    data_directory = os.path.join(cache_dir, 'm4')
    train_directory = os.path.join(data_directory, 'Train')
    test_directory = os.path.join(data_directory, 'Test')

    train_dfs = []
    test_dfs = []

    for id_, dataset_names in files.items():
        file_name = INPUT_MAPPINGS[id_]["file"]
        _download_and_cache(f'Train/{file_name}-train.csv', cache_dir)
        _download_and_cache(f'Test/{file_name}-test.csv', cache_dir)

        # freq = info[info['M4id'] == dataset_name]['Frequency'][0]
        infos = info[info['M4id'].isin(dataset_names)]

        train_df = pd.read_csv(os.path.join(train_directory, f'{file_name}-train.csv'))
        train_df = train_df[train_df['V1'].isin(dataset_names)]
        test_df = pd.read_csv(os.path.join(test_directory, f'{file_name}-test.csv'))
        test_df = test_df[test_df['V1'].isin(dataset_names)]

        for name in dataset_names:
            sp = infos[infos['M4id'] == name]['SP'].iloc[0][0]
            start_date = infos[infos['M4id'] == name]['StartingDate'].iloc[0]

            train = train_df[train_df['V1'] == name].iloc[:, 1:].T
            train = train.dropna().squeeze()
            train.index = pd.period_range(start=start_date, periods=len(train), freq=sp)
            train.name = name
            train_dfs.append(train)

            test = test_df[test_df['V1'] == name].iloc[:, 1:].T
            test = test.dropna().squeeze()
            test.index = pd.period_range(start=train.index[-1] + 1, periods=len(test), freq=sp)
            test.name = name
            test_dfs.append(test)

    if isinstance(dataset_name, str):
        return train_dfs[0], test_dfs[0]
    else:
        return train_dfs, test_dfs


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
