import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List

import dataclasses

from autosktime.data import DatasetProperties
from autosktime.data.benchmark import PHME20Benchmark, CMAPSS1Benchmark, CMAPSS2Benchmark
from autosktime.pipeline.components.base import AutoSktimePreprocessingAlgorithm, COMPONENT_PROPERTIES


@dataclasses.dataclass
class BenchmarkSettings:
    name: str
    sensor_names: List[str]
    setting_names: List[str]
    artificial_names: List[str]
    n_profiles: int


cmapss1 = BenchmarkSettings(
    CMAPSS1Benchmark.name(),
    ['s_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15',
     's_16', 's_17', 's_18', 's_19', 's_20', 's_21'],
    ['setting_1', 'setting_2', 'setting_3'],
    [],
    1
)

cmapss2 = BenchmarkSettings(
    CMAPSS2Benchmark.name(),
    ['s_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15',
     's_16', 's_17', 's_18', 's_19', 's_20', 's_21'],
    ['setting_1', 'setting_2', 'setting_3'],
    [],
    6
)

phme20 = BenchmarkSettings(
    PHME20Benchmark.name(),
    ['Flow_Rate(ml/m)', 'Upstream_Pressure(psi)', 'Downstream_Pressure(psi)'],
    ['Particle Size (micron)', 'Solid Ratio(%)'],
    ['Pressure_Drop'],
    8
)


def find_benchmark_settings(X: pd.DataFrame) -> BenchmarkSettings:
    for settings in filter(lambda o: isinstance(o, BenchmarkSettings), globals().values()):
        if set(settings.setting_names).issubset(X.columns) and set(settings.sensor_names).issubset(X.columns):
            if settings.name.startswith('cmapss_'):
                # All cmapss data sets have the exact same columns. Use operation conditions to select correct settings
                if (X['setting_3'].unique().shape[0] == 1 and settings.n_profiles == 1) or \
                        (X['setting_3'].unique().shape[0] > 1 and settings.n_profiles > 1):
                    return settings
            else:
                return settings
    else:
        raise ValueError(f'Unable to find benchmark settings for columns {X.columns}')


class KMeansOperationCondition(AutoSktimePreprocessingAlgorithm):

    def fit(self, X: pd.DataFrame, y: pd.Series):
        settings = find_benchmark_settings(X)

        sample_df = X.sample(frac=0.05)
        X_train_for_kmeans = sample_df[settings.setting_names].values
        estimator = KMeans(n_clusters=settings.n_profiles, max_iter=10, random_state=self.random_state)
        estimator.fit_predict(X_train_for_kmeans)
        self.estimator = estimator

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        settings = find_benchmark_settings(X)
        X = X.copy()
        # noinspection PyUnresolvedReferences
        X['Kmeans_Profile'] = self.estimator.predict(X[settings.setting_names].values)
        return X

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        pass


class DataScaler(AutoSktimePreprocessingAlgorithm):

    def __init__(self):
        super().__init__()
        self.profiles = []
        self.scalers = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        settings = find_benchmark_settings(X)

        self.scalers = {}
        self.profiles = X['Kmeans_Profile'].unique()
        # Full dataset fit
        for profile in self.profiles:
            sensors_readings = X[(X['Kmeans_Profile'] == profile)].filter(
                settings.sensor_names + settings.artificial_names)  # Get sensor readings
            state_scaler = StandardScaler().fit(sensors_readings)  # Fit scaler
            self.scalers[profile] = state_scaler  # Add to sclaer_list for further reference

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        settings = find_benchmark_settings(X)

        # Full dataset transform
        for profile in self.profiles:
            sensors_readings = X[(X['Kmeans_Profile'] == profile)].filter(
                settings.sensor_names + settings.artificial_names)  # Get sensor readings
            if sensors_readings.shape[0] == 0:
                continue  # No matching profiles found in X_df
            cols = sensors_readings.columns
            normalized_sensor_readings = self.scalers[profile].transform(sensors_readings)  # transform sensor readings
            X.loc[(X['Kmeans_Profile'] == profile), cols] = normalized_sensor_readings  # record transformed values

        return X

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        pass
