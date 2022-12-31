import pandas as pd
from typing import List

import dataclasses

from autosktime.data.benchmark import PHME20Benchmark, CMAPSS1Benchmark, CMAPSS2Benchmark


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
