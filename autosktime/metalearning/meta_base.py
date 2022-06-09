import glob
import logging
import math
import os.path
import pathlib
from typing import List, SupportsFloat, Union, Optional

import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration

from autosktime.constants import TASK_TYPES_TO_STRING
from autosktime.metalearning.kND import KNearestDataSets


class MetaBase:
    def __init__(
            self,
            configuration_space: ConfigurationSpace,
            task: int,
            metric: str,
            distance_measure: str = 'ddtw',
            base_dir: str = None,
            logger: logging.Logger = None
    ):
        self.configuration_space = configuration_space
        self.task = task
        self.metric = metric
        self.distance_measure = distance_measure

        if base_dir is None:
            base_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), 'files')
        self.base_dir = base_dir

        if logger is None:
            logger = logging.Logger('meta-base')
        self.logger = logger

        self._kND: Optional[KNearestDataSets] = None
        self._timeseries: pd.DataFrame = self._load_instances()

    def _load_instances(self) -> pd.DataFrame:
        # TODO store only combined results on disk?
        data_files = glob.glob(
            os.path.join(self.base_dir, f'{TASK_TYPES_TO_STRING[self.task]}_{self.metric}', '*.npy.gz')
        )

        data = {}
        for file in data_files:
            name = pathlib.Path(file).name.split('.')[0]
            series = pd.read_pickle(file)
            data[name] = series

        return pd.DataFrame(data)

    def suggest_configs(
            self,
            y: pd.Series,
            num_initial_configurations: int,
            exclude_double_configurations: bool = True
    ) -> List[Configuration]:
        """Return a list of the best hyperparameters of neighboring datasets"""
        neighbors = self._get_neighbors(y)

        added_configurations = set()
        configurations = []
        for neighbor in neighbors:
            try:
                config = self.get_configuration(neighbor)
                if not exclude_double_configurations or config not in added_configurations:
                    added_configurations.add(config)
                    configurations.append(config)
            except KeyError:
                self.logger.warning(f'Best configuration for {neighbor} not found')
                continue

        return configurations[:num_initial_configurations]

    def _get_neighbors(self, y: pd.Series) -> List[str]:
        if self._kND is None:
            self._kND = KNearestDataSets(metric=self.distance_measure, logger=self.logger)
            self._kND.fit(self._timeseries)

        names, distances, idx = self._kND.kneighbors(y, k=-1)
        return names

    def get_configuration(
            self,
            dataset: str,
            index: Union[int, float] = 0
    ) -> Union[Configuration, List[Configuration]]:
        file = os.path.join(self.base_dir, f'{TASK_TYPES_TO_STRING[self.task]}_{self.metric}', f'{dataset}.csv')
        df = pd.read_csv(file)

        if isinstance(index, float):
            raise ValueError('fractional index not supported yet')

        config_dict = df.loc[index, df.columns.difference(['id', 'train_score', 'test_score'])].to_dict()
        config_dict = {key: value for key, value in config_dict.items()
                       if not isinstance(value, SupportsFloat) or not math.isnan(value)}

        return Configuration(self.configuration_space, config_dict)
