import pandas as pd
from typing import List

import dataclasses

from autosktime.data.benchmark import PHME20Benchmark


@dataclasses.dataclass
class BenchmarkSettings:
    name: str
    sensor_names: List[str]
    setting_names: List[str]
    artificial_names: List[str]
    n_profiles: int


phme20 = BenchmarkSettings(
    PHME20Benchmark.name(),
    ['Flow_Rate(ml/m)', 'Upstream_Pressure(psi)', 'Downstream_Pressure(psi)'],
    ['Particle Size (micron)', 'Solid Ratio(%)'],
    ['Pressure_Drop'],
    8
)


def find_benchmark_settings(X: pd.DataFrame) -> BenchmarkSettings:
    for settings in [phme20]:
        if set(settings.setting_names).issubset(X.columns) and set(settings.sensor_names).issubset(X.columns):
            return settings
    else:
        raise ValueError(f'Unable to find benchmark settings for columns {X.columns}')
