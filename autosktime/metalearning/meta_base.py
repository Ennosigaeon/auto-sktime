import logging
import math
import os.path
import pathlib
from typing import List, SupportsFloat, Union, Optional, Tuple

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
        self._timeseries, self._configs = self._load_instances()

    def _load_instances(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        timeseries = pd.read_pickle(
            os.path.join(self.base_dir, f'{TASK_TYPES_TO_STRING[self.task]}_{self.metric}', 'timeseries.npy.gz')
        )
        configs = pd.read_csv(
            os.path.join(self.base_dir, f'{TASK_TYPES_TO_STRING[self.task]}_{self.metric}', 'configurations.csv')
        )
        return timeseries, configs

    def suggest_configs(
            self,
            y: pd.Series,
            num_initial_configurations: int,
            exclude_double_configurations: bool = True
    ) -> List[Configuration]:
        neighbors, _ = self._get_neighbors(y)

        added_configurations = set()
        configurations = []
        for neighbor in neighbors:
            try:
                config = self.get_configuration(neighbor, best=0)
                if not exclude_double_configurations or config not in added_configurations:
                    added_configurations.add(config)
                    configurations.append(config)
            except KeyError:
                self.logger.warning(f'Best configuration for {neighbor} not found')
                continue

        return configurations[:num_initial_configurations]

    def _get_neighbors(self, y: pd.Series) -> Tuple[List[str], List[float]]:
        if self._kND is None:
            self._kND = KNearestDataSets(metric=self.distance_measure, logger=self.logger)
            self._kND.fit(self._timeseries)

        names, distances = self._kND.kneighbors(y, k=-1)
        return names, distances

    def get_configuration(
            self,
            dataset: str,
            best: Union[int, float]
    ) -> Union[Configuration, List[Configuration]]:
        dataset_configs = self._configs.loc[
            self._configs['dataset'] == dataset,
            self._configs.columns.difference(['dataset', 'id', 'train_score', 'test_score'])
        ]

        def to_config(row: pd.Series):
            return Configuration(self.configuration_space,
                                 {key: value for key, value in row.to_dict().items()
                                  if not isinstance(value, SupportsFloat) or not math.isnan(value)}
                                 )

        if isinstance(best, int):
            return to_config(dataset_configs.iloc[best, :])
        else:
            index = range(int(best * dataset_configs.shape[0]))
            configs = [to_config(row) for _, row in dataset_configs.iloc[index, :].iterrows()]
            return configs
